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
    
    R: cam2world rotation matrix (= np.transpose(qvec2rotmat(qvec)))
    T: translation vector (world2cam)
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
        self.fov = fov  # Fisheye FOV in degrees
        self.fisheye_params = fisheye_params
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

        # Cubemap 카메라들은 렌더링 시 동적으로 생성 (메모리 절약)
        # create_cubemap_cameras()는 매 렌더링마다 호출
        self._cubemap_cameras = None

    @property
    def cubemap_cameras(self):
        """Lazy creation of cubemap cameras"""
        if self._cubemap_cameras is None:
            self._cubemap_cameras = self.create_cubemap_cameras()
        return self._cubemap_cameras

    def create_cubemap_cameras(self):
        """
        6방향 cubemap 카메라 생성
        
        핵심: self.R은 cam2world rotation.
        각 cubemap face는 fisheye 카메라의 로컬 좌표계에서 특정 방향을 바라봄.
        
        Fisheye 카메라 좌표계:
          - Z축: optical axis (앞)
          - X축: 오른쪽
          - Y축: 아래
          
        Cubemap face 방향 (카메라 로컬 기준):
          +X (right):  카메라 기준 오른쪽
          -X (left):   카메라 기준 왼쪽
          +Y (down):   카메라 기준 아래
          -Y (up):     카메라 기준 위
          +Z (front):  카메라 기준 앞 (optical axis)
          -Z (back):   카메라 기준 뒤
        """
        cubemap_cameras = []
        
        face_size = 1024
        fov = math.pi / 2.0  # 정확히 90도
        
        # 각 cubemap face의 로컬 회전 행렬
        # 이 행렬들은 "identity camera (Z=front)"에서 각 방향을 바라보도록 회전시킴
        # 
        # Convention: 각 face camera의 -Z축이 해당 방향을 가리킴
        # (OpenGL convention: camera looks along -Z)
        #
        # R_face: 카메라 로컬 좌표계에서의 회전
        # 최종 cubemap camera의 R = self.R @ R_face
        
        face_rotations = []
        
        # Face 0: +X (오른쪽을 바라봄)
        # -Z_new = +X_orig → 카메라가 +X 방향을 바라봄
        face_rotations.append(np.array([
            [0, 0, 1],    # X_new = Z_orig
            [0, 1, 0],    # Y_new = Y_orig (아래)
            [-1, 0, 0],   # Z_new = -X_orig → -Z_new = +X_orig
        ], dtype=np.float64))
        
        # Face 1: -X (왼쪽을 바라봄)
        face_rotations.append(np.array([
            [0, 0, -1],   # X_new = -Z_orig
            [0, 1, 0],    # Y_new = Y_orig
            [1, 0, 0],    # Z_new = X_orig → -Z_new = -X_orig
        ], dtype=np.float64))
        
        # Face 2: +Y (위를 바라봄 - 카메라 좌표계에서 +Y는 아래이므로, 이건 "아래를 바라봄")
        # 주의: cubemap에서 +Y는 "위"이지만, 카메라 좌표계에서 Y는 아래
        # 여기서는 fisheye 매핑의 +Y 방향과 일치시킴
        face_rotations.append(np.array([
            [1, 0, 0],    # X_new = X_orig
            [0, 0, 1],    # Y_new = Z_orig
            [0, -1, 0],   # Z_new = -Y_orig → -Z_new = +Y_orig
        ], dtype=np.float64))
        
        # Face 3: -Y (위를 바라봄)
        face_rotations.append(np.array([
            [1, 0, 0],    # X_new = X_orig
            [0, 0, -1],   # Y_new = -Z_orig
            [0, 1, 0],    # Z_new = Y_orig → -Z_new = -Y_orig
        ], dtype=np.float64))
        
        # Face 4: +Z (앞을 바라봄 - identity)
        face_rotations.append(np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float64))
        
        # Face 5: -Z (뒤를 바라봄)
        face_rotations.append(np.array([
            [-1, 0, 0],   # X_new = -X_orig (좌우반전)
            [0, 1, 0],    # Y_new = Y_orig
            [0, 0, -1],   # Z_new = -Z_orig → -Z_new = +Z_orig (뒤)
        ], dtype=np.float64))
        
        face_names = ['pos_x', 'neg_x', 'pos_y', 'neg_y', 'pos_z', 'neg_z']
        
        for i, (R_face, name) in enumerate(zip(face_rotations, face_names)):
            # self.R: cam2world for the fisheye camera
            # R_face: rotation within camera's local frame
            # Combined: R_cubemap = self.R @ R_face (cam2world for cubemap face)
            R_cubemap = self.R @ R_face
            
            # Dummy image (cubemap face는 GT 이미지가 필요없음)
            dummy_image = np.zeros((face_size, face_size, 3), dtype=np.float32)
            
            cam = Camera(
                colmap_id=self.colmap_id * 10 + i,
                R=R_cubemap,
                T=self.T,  # Translation은 동일 (같은 위치에서 촬영)
                FoVx=fov,
                FoVy=fov,
                depth_params={},
                image=dummy_image,
                gt_alpha_mask=None,
                image_name=f"{self.image_name}_cubemap_{name}",
                uid=self.uid * 10 + i,
                trans=self.trans,
                scale=self.scale,
                data_device="cpu",  # dummy image는 CPU에
                train_test_exp=False,
                is_test_dataset=self.is_test_dataset,
                scene_scale=self.scene_scale
            )
            
            cubemap_cameras.append(cam)
        
        return cubemap_cameras


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, camera_center=None):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        if camera_center is not None:
            self.camera_center = camera_center
        else:
            self.camera_center = self.world_view_transform.inverse()[3, :3]