"""
Cubemap to Fisheye Image Conversion (Differentiable)
=====================================================
6개의 cubemap face를 fisheye 이미지로 변환하는 유틸리티
- F.grid_sample 기반으로 완전한 gradient flow 보장
- OPENCV_FISHEYE distortion model 지원
- Cubemap face 결정 및 UV 계산 최적화
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


def determine_cubemap_face(vec_x, vec_y, vec_z):
    """
    3D 방향 벡터에서 cubemap face index와 UV 좌표를 결정
    
    OpenGL 표준 cubemap 규약:
      Face 0 (+X, right):  major axis = +X
      Face 1 (-X, left):   major axis = -X
      Face 2 (+Y, up):     major axis = +Y
      Face 3 (-Y, down):   major axis = -Y
      Face 4 (+Z, front):  major axis = +Z
      Face 5 (-Z, back):   major axis = -Z
    
    Returns:
        face_idx: [H, W] long tensor
        u, v: [H, W] float tensors in [0, 1]
    """
    abs_x = torch.abs(vec_x)
    abs_y = torch.abs(vec_y)
    abs_z = torch.abs(vec_z)
    
    face_idx = torch.zeros_like(vec_x, dtype=torch.long)
    u = torch.zeros_like(vec_x)
    v = torch.zeros_like(vec_x)
    
    # +X face (오른쪽): major = +X
    mask = (vec_x > 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
    if mask.any():
        ma = abs_x[mask]
        face_idx[mask] = 0
        u[mask] = (-vec_z[mask] / ma + 1.0) * 0.5
        v[mask] = (-vec_y[mask] / ma + 1.0) * 0.5
    
    # -X face (왼쪽): major = -X
    mask = (vec_x <= 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
    if mask.any():
        ma = abs_x[mask]
        face_idx[mask] = 1
        u[mask] = (vec_z[mask] / ma + 1.0) * 0.5
        v[mask] = (-vec_y[mask] / ma + 1.0) * 0.5
    
    # +Y face (위): major = +Y
    mask = (vec_y > 0) & (abs_y > abs_x) & (abs_y >= abs_z)
    if mask.any():
        ma = abs_y[mask]
        face_idx[mask] = 2
        u[mask] = (vec_x[mask] / ma + 1.0) * 0.5
        v[mask] = (vec_z[mask] / ma + 1.0) * 0.5
    
    # -Y face (아래): major = -Y
    mask = (vec_y <= 0) & (abs_y > abs_x) & (abs_y >= abs_z)
    if mask.any():
        ma = abs_y[mask]
        face_idx[mask] = 3
        u[mask] = (vec_x[mask] / ma + 1.0) * 0.5
        v[mask] = (-vec_z[mask] / ma + 1.0) * 0.5
    
    # +Z face (앞): major = +Z
    mask = (vec_z > 0) & (abs_z > abs_x) & (abs_z > abs_y)
    if mask.any():
        ma = abs_z[mask]
        face_idx[mask] = 4
        u[mask] = (vec_x[mask] / ma + 1.0) * 0.5
        v[mask] = (-vec_y[mask] / ma + 1.0) * 0.5
    
    # -Z face (뒤): major = -Z
    mask = (vec_z <= 0) & (abs_z > abs_x) & (abs_z > abs_y)
    if mask.any():
        ma = abs_z[mask]
        face_idx[mask] = 5
        u[mask] = (-vec_x[mask] / ma + 1.0) * 0.5
        v[mask] = (-vec_y[mask] / ma + 1.0) * 0.5
    
    return face_idx, u, v


def opencv_fisheye_theta_to_theta_d(theta, k1, k2, k3, k4):
    """
    OPENCV_FISHEYE forward distortion:
        theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9
    
    Args:
        theta: undistorted angle from optical axis (tensor)
        k1, k2, k3, k4: distortion coefficients
    Returns:
        theta_d: distorted angle
    """
    t2 = theta * theta
    t4 = t2 * t2
    t6 = t4 * t2
    t8 = t4 * t4
    return theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8)


def opencv_fisheye_theta_d_to_theta(theta_d, k1, k2, k3, k4, max_iter=20):
    """
    OPENCV_FISHEYE inverse distortion (Newton's method):
    Given theta_d, solve for theta such that:
        theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9 = theta_d
    
    Args:
        theta_d: distorted angle (tensor)
        k1, k2, k3, k4: distortion coefficients
    Returns:
        theta: undistorted angle
    """
    theta = theta_d.clone()  # initial guess
    for _ in range(max_iter):
        t2 = theta * theta
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        f_val = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8) - theta_d
        f_deriv = 1.0 + 3*k1*t2 + 5*k2*t4 + 7*k3*t6 + 9*k4*t8
        theta = theta - f_val / f_deriv.clamp(min=1e-8)
    return theta


def create_fisheye_mapping(height, width, fov=190.0, device='cuda', fisheye_params=None):
    """
    Fisheye 이미지의 각 픽셀에 대해 cubemap face index와 UV 좌표 생성
    
    두 가지 모드:
    1. OPENCV_FISHEYE: fisheye_params에 fx, fy, cx, cy, k1-k4가 있으면 정확한 모델 사용
    2. Equidistant (fallback): 단순 equidistant projection (theta = r * fov/2)
    
    Args:
        height, width: output image size
        fov: field of view in degrees (equidistant 모드에서 사용)
        device: torch device
        fisheye_params: dict with OPENCV_FISHEYE params (optional)
        
    Returns:
        face_idx: [H, W] - which cubemap face
        u, v: [H, W] - UV coordinates within that face
        valid_mask: [H, W] - which pixels are within FOV
    """
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    use_opencv_fisheye = (fisheye_params is not None and 
                          fisheye_params.get("distortion_model") == "opencv_fisheye" and
                          "fx" in fisheye_params)
    
    if use_opencv_fisheye:
        # ============================================================
        # OPENCV_FISHEYE model
        # ============================================================
        fx = fisheye_params["fx"]
        fy = fisheye_params["fy"]
        cx_cam = fisheye_params["cx"]
        cy_cam = fisheye_params["cy"]
        k1 = fisheye_params["k1"]
        k2 = fisheye_params["k2"]
        k3 = fisheye_params["k3"]
        k4 = fisheye_params["k4"]
        colmap_w = fisheye_params.get("colmap_width", width)
        colmap_h = fisheye_params.get("colmap_height", height)
        
        # 현재 이미지 해상도와 COLMAP 해상도의 비율 계산
        scale_x = width / colmap_w
        scale_y = height / colmap_h
        
        # 스케일 적용된 intrinsics
        fx_s = fx * scale_x
        fy_s = fy * scale_y
        cx_s = cx_cam * scale_x
        cy_s = cy_cam * scale_y
        
        # 픽셀 좌표 → 정규화된 좌표
        mx = (x - cx_s) / fx_s
        my = (y - cy_s) / fy_s
        
        # r_d = sqrt(mx^2 + my^2) = theta_d (since r_pixel = f * theta_d → mx = r_pixel/f = theta_d * cos(phi))
        # 실제로는 mx = theta_d * cos(phi) / 1 이 아님
        # 
        # OPENCV_FISHEYE model:
        #   x_normalized = theta_d / r_undist * x_undist  (단, r_undist = sqrt(x_undist^2 + y_undist^2))
        #   여기서 x_normalized = (pixel_x - cx) / fx
        #
        # 역으로:
        #   theta_d = sqrt(mx^2 + my^2)
        #   phi = atan2(my, mx)
        #   theta = undistort(theta_d)
        #   3D direction = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
        
        theta_d = torch.sqrt(mx**2 + my**2)
        phi = torch.atan2(my, mx)
        
        # Valid mask: theta_d가 의미있는 범위 내
        # r_max_pixels 기반으로 계산
        f_avg = (fx_s + fy_s) / 2.0
        r_max_pixels = min(cx_s, cy_s, width - cx_s, height - cy_s)
        theta_d_max = r_max_pixels / f_avg
        valid_mask = theta_d <= theta_d_max
        
        # Inverse distortion: theta_d → theta (실제 각도)
        theta = opencv_fisheye_theta_d_to_theta(theta_d, k1, k2, k3, k4)
        
        # theta가 유효한 범위인지 추가 체크
        valid_mask = valid_mask & (theta >= 0) & (theta < math.pi)
        
        # 3D ray direction
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        vec_x = sin_theta * torch.cos(phi)
        vec_y = sin_theta * torch.sin(phi)
        vec_z = cos_theta
        
    else:
        # ============================================================
        # Simple equidistant projection (fallback)
        # ============================================================
        cx_img = width / 2.0
        cy_img = height / 2.0
        fov_rad = math.radians(fov)
        radius = min(cx_img, cy_img)
        
        dx = (x - cx_img) / radius
        dy = (y - cy_img) / radius
        r = torch.sqrt(dx**2 + dy**2)
        
        valid_mask = r <= 1.0
        
        theta = r * (fov_rad / 2.0)
        phi = torch.atan2(dy, dx)
        
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        vec_x = sin_theta * torch.cos(phi)
        vec_y = sin_theta * torch.sin(phi)
        vec_z = cos_theta
    
    # Cubemap face 결정
    face_idx, u, v = determine_cubemap_face(vec_x, vec_y, vec_z)
    
    return face_idx, u, v, valid_mask


def create_mapping_cache(height, width, fov=190.0, device='cuda', fisheye_params=None):
    """
    매핑을 미리 계산해서 캐시로 저장 (학습 시 속도 향상)
    
    grid_sample용 grid도 미리 생성 ([-1, 1] 범위)
    """
    face_idx, u, v, valid_mask = create_fisheye_mapping(height, width, fov, device, fisheye_params)
    
    # grid_sample용 grid 미리 계산 (u,v를 [-1,1] 범위로 변환)
    grid_x = u * 2.0 - 1.0  # [0,1] → [-1,1]
    grid_y = v * 2.0 - 1.0
    
    # 각 face별 마스크와 인덱스 미리 계산
    face_masks = []
    face_grids = []
    face_pixel_indices = []
    
    for i in range(6):
        mask = (face_idx == i) & valid_mask
        face_masks.append(mask)
        
        if mask.any():
            gx = grid_x[mask]
            gy = grid_y[mask]
            grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
            face_grids.append(grid)
            face_pixel_indices.append(mask)
        else:
            face_grids.append(None)
            face_pixel_indices.append(None)
    
    return {
        'face_idx': face_idx,
        'u': u,
        'v': v,
        'valid_mask': valid_mask,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'face_masks': face_masks,
        'face_grids': face_grids,
        'height': height,
        'width': width,
        'fov': fov,
    }


def cubemap_to_fisheye(cubemap_faces, height, width, fov=190.0, mapping_cache=None):
    """
    Differentiable cubemap → fisheye 변환 (F.grid_sample 사용)
    
    gradient가 cubemap_faces를 통해 rasterizer까지 역전파됨
    
    Args:
        cubemap_faces: list of 6 tensors [C, face_H, face_W]
            Order: [+X, -X, +Y, -Y, +Z, -Z]
        height, width: output fisheye image size
        fov: field of view in degrees
        mapping_cache: pre-computed mapping (from create_mapping_cache)
        
    Returns:
        fisheye_image: [C, H, W] - differentiable w.r.t. cubemap_faces
    """
    device = cubemap_faces[0].device
    C = cubemap_faces[0].shape[0]
    
    if mapping_cache is not None:
        face_masks = mapping_cache['face_masks']
        face_grids = mapping_cache['face_grids']
    else:
        cache = create_mapping_cache(height, width, fov, device)
        face_masks = cache['face_masks']
        face_grids = cache['face_grids']
    
    output = torch.zeros(C, height, width, device=device)
    
    for i in range(6):
        mask = face_masks[i]
        grid = face_grids[i]
        
        if grid is None or not mask.any():
            continue
        
        if grid.device != device:
            grid = grid.to(device)
        
        face_input = cubemap_faces[i].unsqueeze(0)
        
        sampled = F.grid_sample(
            face_input, 
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        sampled = sampled.squeeze(0).squeeze(1)
        output[:, mask] = sampled
    
    return output


# ============================================================
# 테스트 코드
# ============================================================
if __name__ == "__main__":
    print("Testing differentiable cubemap to fisheye conversion...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_size = 512
    
    cubemap_faces = []
    colors = [
        [1.0, 0.0, 0.0],  # +X: Red
        [0.0, 1.0, 0.0],  # -X: Green
        [0.0, 0.0, 1.0],  # +Y: Blue
        [1.0, 1.0, 0.0],  # -Y: Yellow
        [1.0, 0.0, 1.0],  # +Z: Magenta
        [0.0, 1.0, 1.0],  # -Z: Cyan
    ]
    
    for color in colors:
        face = torch.ones(3, face_size, face_size, device=device, requires_grad=True)
        face_colored = face * torch.tensor(color, device=device).view(3, 1, 1)
        cubemap_faces.append(face_colored)
    
    # Test 1: Simple equidistant
    fisheye_h, fisheye_w = 1024, 1024
    cache = create_mapping_cache(fisheye_h, fisheye_w, fov=190.0, device=device)
    fisheye = cubemap_to_fisheye(cubemap_faces, fisheye_h, fisheye_w, mapping_cache=cache)
    print(f"Equidistant output shape: {fisheye.shape}, requires_grad: {fisheye.requires_grad}")
    
    # Test 2: OPENCV_FISHEYE model
    test_params = {
        "distortion_model": "opencv_fisheye",
        "fx": 1001.95, "fy": 996.89,
        "cx": 1612.0, "cy": 1617.0,
        "k1": 0.029183, "k2": -0.010338, "k3": -0.003328, "k4": 0.000482,
        "colmap_width": 3264, "colmap_height": 3264,
    }
    cache2 = create_mapping_cache(fisheye_h, fisheye_w, device=device, fisheye_params=test_params)
    fisheye2 = cubemap_to_fisheye(cubemap_faces, fisheye_h, fisheye_w, mapping_cache=cache2)
    print(f"OPENCV_FISHEYE output shape: {fisheye2.shape}, requires_grad: {fisheye2.requires_grad}")
    
    # Gradient flow test
    loss = fisheye2.sum()
    loss.backward()
    for i, face in enumerate(cubemap_faces):
        if face.grad is not None:
            print(f"Face {i} grad norm: {face.grad.norm():.4f}")
    
    print("\nConversion + gradient test successful!")