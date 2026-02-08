#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_fisheye
from scene.cameras import FisheyeCamera
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.cubemap_to_fisheye import create_mapping_cache


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, fisheye_cache=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        is_fisheye = isinstance(view, FisheyeCamera)
        
        if is_fisheye:
            cache_key = f"{view.image_height}_{view.image_width}_{view.fov}"
            if fisheye_cache is None:
                fisheye_cache = {}
            if cache_key not in fisheye_cache:
                fisheye_cache[cache_key] = create_mapping_cache(
                    view.image_height, view.image_width,
                    fov=view.fov, device='cuda'
                )
            rendering = render_fisheye(view, gaussians, pipeline, background, 
                                       mapping_cache=fisheye_cache[cache_key])["render"]
        else:
            rendering = render(view, gaussians, pipeline, background)["render"]
        
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, 
                skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        fisheye_cache = {}

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, 
                      scene.getTrainCameras(), gaussians, pipeline, background, fisheye_cache)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, 
                      scene.getTestCameras(), gaussians, pipeline, background, fisheye_cache)


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), 
                args.skip_train, args.skip_test)