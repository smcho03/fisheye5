#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import uuid
import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, render_fisheye, network_gui
import sys
from scene import Scene, GaussianModel
from scene.cameras import FisheyeCamera
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.cubemap_to_fisheye import create_mapping_cache

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def add_fisheye_densification_stats(gaussians, viewspace_points_list, visibility_filter):
    """
    Fisheye 렌더링에서 6개 cubemap face의 viewspace gradient를 통합하여 densification stats에 추가
    
    각 face에서 관측된 gradient를 합산하고, 관측 횟수로 나누어 평균 gradient를 사용
    """
    n_points = gaussians.get_xyz.shape[0]
    
    # 모든 face의 gradient를 합산
    combined_grad = torch.zeros(n_points, 2, device="cuda")
    combined_count = torch.zeros(n_points, 1, device="cuda")
    
    for viewspace_points in viewspace_points_list:
        if viewspace_points.grad is not None:
            grad = viewspace_points.grad[:, :2]  # xy gradient만
            # 유효한 gradient가 있는 점만
            valid = (grad.abs().sum(dim=-1) > 0)
            if valid.any():
                combined_grad[valid] += grad[valid]
                combined_count[valid] += 1
    
    # 관측된 점에 대해 평균 gradient 계산
    observed = (combined_count > 0).squeeze()
    if observed.any():
        # gradient norm 계산 (합산된 gradient 사용)
        grad_norm = torch.norm(combined_grad[observed], dim=-1, keepdim=True)
        
        # densification stats에 추가
        gaussians.xyz_gradient_accum[observed] += grad_norm
        gaussians.denom[observed] += 1


def training(dataset, opt, pipe, testing_iterations, saving_iterations, 
             checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # Fisheye 매핑 캐시 생성 (한 번만 계산)
    fisheye_mapping_cache = {}
    
    for iteration in range(first_iter, opt.iterations + 1):
        # Network GUI
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # ============================================================
        # Fisheye vs Pinhole 렌더링
        # ============================================================
        is_fisheye = isinstance(viewpoint_cam, FisheyeCamera)
        
        if is_fisheye:
            # Fisheye: cubemap → fisheye (differentiable)
            cache_key = f"{viewpoint_cam.image_height}_{viewpoint_cam.image_width}_{viewpoint_cam.fov}"
            if cache_key not in fisheye_mapping_cache:
                fisheye_mapping_cache[cache_key] = create_mapping_cache(
                    viewpoint_cam.image_height,
                    viewpoint_cam.image_width,
                    fov=viewpoint_cam.fov,
                    device='cuda'
                )
            
            render_pkg = render_fisheye(
                viewpoint_cam, gaussians, pipe, bg,
                mapping_cache=fisheye_mapping_cache[cache_key]
            )
            
            image = render_pkg["render"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            viewspace_points_list = render_pkg["viewspace_points_list"]
            
        else:
            # Pinhole: 일반 렌더링
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            
            image = render_pkg["render"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            viewspace_point_tensor = render_pkg["viewspace_points"]

        # ============================================================
        # Loss 계산
        # ============================================================
        gt_image = viewpoint_cam.original_image.cuda()
        
        # Exposure compensation
        if viewpoint_cam.exposure is not None:
            exposure = viewpoint_cam.exposure
            image = image * torch.exp(exposure[0]) + exposure[1:].unsqueeze(-1).unsqueeze(-1)
        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # Depth regularization (pinhole only - fisheye는 cubemap 단계에서는 미적용)
        if not is_fisheye and hasattr(viewpoint_cam, 'depth_image') and viewpoint_cam.depth_image is not None:
            if "depth" in render_pkg:
                depth_loss = l1_loss(render_pkg["depth"], viewpoint_cam.depth_image)
                loss = loss + 0.1 * depth_loss
        
        loss.backward()

        iter_end.record()

        # ============================================================
        # Debug 저장 (1000 iteration마다)
        # ============================================================
        if is_fisheye and iteration % 1000 == 0:
            try:
                from utils.debug_fisheye import save_fisheye_comparison
                debug_base = os.path.join(dataset.model_path, f"debug_iter_{iteration}")
                os.makedirs(debug_base, exist_ok=True)
                comparison_path = os.path.join(debug_base, "fisheye_comparison.png")
                save_fisheye_comparison(image.detach(), gt_image, comparison_path)
            except Exception as e:
                print(f"Debug save failed: {e}")

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, 
                          iter_start.elapsed_time(iter_end), testing_iterations, 
                          scene, render, (pipe, background),
                          fisheye_cache=fisheye_mapping_cache)
            
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # ============================================================
            # Densification
            # ============================================================
            if iteration < opt.densify_until_iter:
                # max_radii2D 업데이트
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], 
                    radii[visibility_filter]
                )
                
                # Densification stats 추가
                if is_fisheye:
                    # Fisheye: 6개 face의 viewspace gradient를 통합
                    add_fisheye_densification_stats(
                        gaussians, viewspace_points_list, visibility_filter
                    )
                else:
                    # Pinhole: 기존 방식
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 
                                                scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or \
                   (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                
                # Exposure optimizer
                if viewpoint_cam.exposure is not None:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), 
                          scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, 
                   testing_iterations, scene: Scene, renderFunc, renderArgs,
                   fisheye_cache=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()}, 
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    cam_is_fisheye = isinstance(viewpoint, FisheyeCamera)
                    
                    if cam_is_fisheye:
                        cache_key = f"{viewpoint.image_height}_{viewpoint.image_width}_{viewpoint.fov}"
                        if fisheye_cache is None or cache_key not in fisheye_cache:
                            cache = create_mapping_cache(
                                viewpoint.image_height,
                                viewpoint.image_width,
                                fov=viewpoint.fov,
                                device='cuda'
                            )
                        else:
                            cache = fisheye_cache[cache_key]
                        
                        image = render_fisheye(viewpoint, scene.gaussians, *renderArgs, mapping_cache=cache)["render"]
                    else:
                        image = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"]
                    
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if viewpoint.exposure is not None:
                        exposure = viewpoint.exposure
                        image = image * torch.exp(exposure[0]) + exposure[1:].unsqueeze(-1).unsqueeze(-1)
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--disable_viewer", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), 
             args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")