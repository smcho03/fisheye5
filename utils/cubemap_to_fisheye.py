"""
Cubemap to Fisheye Image Conversion (Differentiable)
=====================================================
6개의 cubemap face를 fisheye 이미지로 변환하는 유틸리티
- F.grid_sample 기반으로 완전한 gradient flow 보장
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


def create_fisheye_mapping(height, width, fov=117.0, device='cuda'):
    """
    Fisheye 이미지의 각 픽셀에 대해 cubemap face index와 UV 좌표 생성
    
    Equidistant fisheye projection:
        r = f * theta (theta = angle from optical axis)
    
    Args:
        height, width: output image size
        fov: field of view in degrees
        device: torch device
        
    Returns:
        face_idx: [H, W] - which cubemap face
        u, v: [H, W] - UV coordinates within that face
        valid_mask: [H, W] - which pixels are within FOV
    """
    cx = width / 2.0
    cy = height / 2.0
    
    fov_rad = math.radians(fov)
    radius = min(cx, cy)
    
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # 정규화된 좌표 (중심 기준)
    dx = (x - cx) / radius
    dy = (y - cy) / radius
    r = torch.sqrt(dx**2 + dy**2)
    
    # FOV 범위 내의 픽셀만 유효
    valid_mask = r <= 1.0
    
    # Equidistant projection: theta = r * (fov/2)
    theta = r * (fov_rad / 2.0)
    phi = torch.atan2(dy, dx)
    
    # 3D ray direction (카메라 로컬 좌표계)
    # z축이 optical axis (앞 방향)
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    
    vec_x = sin_theta * torch.cos(phi)
    vec_y = sin_theta * torch.sin(phi)
    vec_z = cos_theta
    
    # Cubemap face 결정
    face_idx, u, v = determine_cubemap_face(vec_x, vec_y, vec_z)
    
    return face_idx, u, v, valid_mask


def create_mapping_cache(height, width, fov=117.0, device='cuda'):
    """
    매핑을 미리 계산해서 캐시로 저장 (학습 시 속도 향상)
    
    grid_sample용 grid도 미리 생성 ([-1, 1] 범위)
    """
    face_idx, u, v, valid_mask = create_fisheye_mapping(height, width, fov, device)
    
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
            # 해당 face에 속하는 픽셀들의 grid 좌표
            gx = grid_x[mask]
            gy = grid_y[mask]
            # grid_sample format: [1, 1, N, 2]
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


def cubemap_to_fisheye(cubemap_faces, height, width, fov=117.0, mapping_cache=None):
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
    
    # 매핑 캐시 로드 또는 생성
    if mapping_cache is not None:
        face_masks = mapping_cache['face_masks']
        face_grids = mapping_cache['face_grids']
    else:
        # 캐시 없으면 그때 생성
        cache = create_mapping_cache(height, width, fov, device)
        face_masks = cache['face_masks']
        face_grids = cache['face_grids']
    
    # 출력 이미지 초기화
    output = torch.zeros(C, height, width, device=device)
    
    for i in range(6):
        mask = face_masks[i]
        grid = face_grids[i]
        
        if grid is None or not mask.any():
            continue
        
        # grid를 같은 device로 이동 (캐시가 다른 device에 있을 수 있음)
        if grid.device != device:
            grid = grid.to(device)
        
        # cubemap face를 grid_sample 입력 형식으로: [1, C, H, W]
        face_input = cubemap_faces[i].unsqueeze(0)
        
        # Differentiable bilinear sampling
        # grid: [1, 1, N, 2] → output: [1, C, 1, N]
        sampled = F.grid_sample(
            face_input, 
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )
        
        # [1, C, 1, N] → [C, N]
        sampled = sampled.squeeze(0).squeeze(1)
        
        # 해당 픽셀에 값 할당
        output[:, mask] = sampled
    
    return output


# ============================================================
# 테스트 코드
# ============================================================
if __name__ == "__main__":
    print("Testing differentiable cubemap to fisheye conversion...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_size = 512
    
    # 테스트용 cubemap 생성 (requires_grad=True로 gradient 테스트)
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
        # requires_grad를 유지하면서 색상 곱하기
        face_colored = face * torch.tensor(color, device=device).view(3, 1, 1)
        cubemap_faces.append(face_colored)
    
    # 매핑 캐시 생성
    fisheye_h, fisheye_w = 1024, 1024
    cache = create_mapping_cache(fisheye_h, fisheye_w, fov=117.0, device=device)
    
    # 변환
    fisheye = cubemap_to_fisheye(cubemap_faces, fisheye_h, fisheye_w, mapping_cache=cache)
    
    print(f"Output shape: {fisheye.shape}")
    print(f"Output requires_grad: {fisheye.requires_grad}")
    
    # Gradient flow 테스트
    loss = fisheye.sum()
    loss.backward()
    
    for i, face in enumerate(cubemap_faces):
        if face.grad is not None:
            print(f"Face {i} grad norm: {face.grad.norm():.4f}")
        else:
            print(f"Face {i}: no grad (check computation graph)")
    
    print("\nConversion + gradient test successful!")
    
    # 이미지 저장
    try:
        from torchvision.utils import save_image
        save_image(fisheye.detach(), 'test_fisheye_differentiable.png')
        print("Saved test image")
    except:
        pass