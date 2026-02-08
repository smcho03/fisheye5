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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.cubemap_to_fisheye import cubemap_to_fisheye, create_mapping_cache


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
           scaling_modifier=1.0, override_color=None, return_depth=False, return_opacity=False):
    """
    Render the scene (standard pinhole camera).
    """
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    rendered_image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    result = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }

    if return_depth:
        result["depth"] = depth

    return result


def render_fisheye(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                   scaling_modifier=1.0, override_color=None, mapping_cache=None):
    """
    Fisheye 카메라 렌더링 (Cubemap → Fisheye, fully differentiable)
    
    mapping_cache가 OPENCV_FISHEYE distortion model을 포함할 수 있음
    """
    from scene.cameras import FisheyeCamera
    
    if not isinstance(viewpoint_camera, FisheyeCamera):
        raise ValueError("render_fisheye requires FisheyeCamera instance")
    
    cubemap_cameras = viewpoint_camera.cubemap_cameras
    
    cubemap_faces = []
    all_viewspace_points = []
    all_visibility_filters = []
    all_radii = []
    
    # 1. 6개 방향 각각 렌더링
    for i, cam in enumerate(cubemap_cameras):
        result = render(cam, pc, pipe, bg_color, scaling_modifier, override_color)
        
        cubemap_faces.append(result["render"])
        all_viewspace_points.append(result["viewspace_points"])
        all_visibility_filters.append(result["visibility_filter"])
        all_radii.append(result["radii"])
    
    # 2. Differentiable cubemap → fisheye 변환
    fisheye_image = cubemap_to_fisheye(
        cubemap_faces,
        height=viewpoint_camera.image_height,
        width=viewpoint_camera.image_width,
        fov=viewpoint_camera.fov,
        mapping_cache=mapping_cache
    )
    
    # 3. Visibility와 radii 통합
    combined_visibility = all_visibility_filters[0]
    for vis in all_visibility_filters[1:]:
        combined_visibility = combined_visibility | vis
    
    combined_radii = all_radii[0]
    for rad in all_radii[1:]:
        combined_radii = torch.maximum(combined_radii, rad)
    
    return {
        "render": fisheye_image,
        "viewspace_points_list": all_viewspace_points,
        "visibility_filter": combined_visibility,
        "radii": combined_radii,
        "cubemap_faces": cubemap_faces,
    }