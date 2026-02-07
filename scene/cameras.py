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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import math


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, depth_params, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, scene_scale = 1.0,
                 depth_image=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.depth_params = depth_params
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        if isinstance(image, torch.Tensor):
            resized_image_rgb = image if image.shape[0] == 3 else image.permute(2, 0, 1)
        else:
            resized_image_rgb = torch.from_numpy(image).permute(2, 0, 1)
        self.original_image = resized_image_rgb.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if train_test_exp:
            self.exposure = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], device=self.data_device, requires_grad=True))
        else:
            self.exposure = None

        self.is_test_dataset = is_test_dataset
        self.scene_scale = scene_scale

        if depth_image is not None:
            self.depth_image = torch.from_numpy(depth_image).to(self.data_device)
        else:
            self.depth_image = None


class FisheyeCamera(nn.Module):
    """
    Fisheye 카메라 클래스
    - Fisheye GT 이미지를 저장
    - Cubemap 렌더링 후 Fisheye로 변환하여 비교
    """
    def __init__(self, colmap_id, R, T, fov, fisheye_params, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 train_test_exp=False, is_test_dataset=False, scene_scale=1.0,
                 depth_image=None):
        super(FisheyeCamera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.fov = fov  # Fisheye FOV (e.g., 180.0)
        self.fisheye_params = fisheye_params  # Distortion parameters
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        # Fisheye GT 이미지
        if isinstance(image, torch.Tensor):
            resized_image_rgb = image if image.shape[0] == 3 else image.permute(2, 0, 1)
        else:
            resized_image_rgb = torch.from_numpy(image).permute(2, 0, 1)
        self.original_image = resized_image_rgb.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # World view transform (fisheye 카메라 위치)
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # Exposure compensation
        if train_test_exp:
            self.exposure = nn.Parameter(torch.tensor([0.0, 0.0, 0.0], device=self.data_device, requires_grad=True))
        else:
            self.exposure = None

        self.is_test_dataset = is_test_dataset
        self.scene_scale = scene_scale

        if depth_image is not None:
            self.depth_image = torch.from_numpy(depth_image).to(self.data_device)
        else:
            self.depth_image = None

        # Cubemap 카메라들 생성 (6개 방향)
        self.cubemap_cameras = self.create_cubemap_cameras()
        
        
        
    

    def create_cubemap_cameras(self):
        """OpenGL 표준 cubemap 방향"""
        cubemap_cameras = []
        
        # 고정된 크기와 FOV
        face_size = 1024
        fov = math.pi / 2.0  # 정확히 90도
        
        # OpenGL standard cubemap rotations
        rotations = [
            # +X (right)
            np.array([[0, 0, -1],
                    [0, -1, 0],
                    [-1, 0, 0]]),
            
            # -X (left)
            np.array([[0, 0, 1],
                    [0, -1, 0],
                    [1, 0, 0]]),
            
            # +Y (up)
            np.array([[1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0]]),
            
            # -Y (down)
            np.array([[1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]]),
            
            # +Z (forward)
            np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]]),
            
            # -Z (back)
            np.array([[-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]]),
        ]
        
        face_names = ['right', 'left', 'up', 'down', 'front', 'back']
        
        for i, (rot, name) in enumerate(zip(rotations, face_names)):
            R_cubemap = self.R @ rot
            
            dummy_image = np.zeros((face_size, face_size, 3))
            
            cam = Camera(
                colmap_id=self.colmap_id * 10 + i,
                R=R_cubemap,
                T=self.T,
                FoVx=fov,
                FoVy=fov,
                depth_params={},
                image=dummy_image,
                gt_alpha_mask=None,
                image_name=f"{self.image_name}_cubemap_{name}",
                uid=self.uid * 10 + i,
                trans=self.trans,
                scale=self.scale,
                data_device="cpu",
                train_test_exp=False,
                is_test_dataset=self.is_test_dataset,
                scene_scale=self.scene_scale
            )
            
            cubemap_cameras.append(cam)
        
        return cubemap_cameras


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, camera_center):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center