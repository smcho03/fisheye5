"""
Cubemap to Fisheye Image Conversion
====================================
6개의 cubemap face를 fisheye 이미지로 변환하는 유틸리티
"""

import torch
import numpy as np
import math


def determine_cubemap_face(vec_x, vec_y, vec_z):
    """정확한 cubemap face 결정 및 UV 계산"""
    abs_x = torch.abs(vec_x)
    abs_y = torch.abs(vec_y)
    abs_z = torch.abs(vec_z)
    
    face_idx = torch.zeros_like(vec_x, dtype=torch.long)
    u = torch.zeros_like(vec_x)
    v = torch.zeros_like(vec_x)
    
    # +X face (오른쪽)
    mask = (vec_x > 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
    face_idx[mask] = 0
    u[mask] = (-vec_z[mask] / abs_x[mask] + 1.0) * 0.5
    v[mask] = (-vec_y[mask] / abs_x[mask] + 1.0) * 0.5
    
    # -X face (왼쪽)
    mask = (vec_x < 0) & (abs_x >= abs_y) & (abs_x >= abs_z)
    face_idx[mask] = 1
    u[mask] = (vec_z[mask] / abs_x[mask] + 1.0) * 0.5
    v[mask] = (-vec_y[mask] / abs_x[mask] + 1.0) * 0.5
    
    # +Y face (위)
    mask = (vec_y > 0) & (abs_y >= abs_x) & (abs_y >= abs_z)
    face_idx[mask] = 2
    u[mask] = (vec_x[mask] / abs_y[mask] + 1.0) * 0.5
    v[mask] = (vec_z[mask] / abs_y[mask] + 1.0) * 0.5
    
    # -Y face (아래)
    mask = (vec_y < 0) & (abs_y >= abs_x) & (abs_y >= abs_z)
    face_idx[mask] = 3
    u[mask] = (vec_x[mask] / abs_y[mask] + 1.0) * 0.5
    v[mask] = (-vec_z[mask] / abs_y[mask] + 1.0) * 0.5
    
    # +Z face (앞)
    mask = (vec_z > 0) & (abs_z >= abs_x) & (abs_z >= abs_y)
    face_idx[mask] = 4
    u[mask] = (vec_x[mask] / abs_z[mask] + 1.0) * 0.5
    v[mask] = (-vec_y[mask] / abs_z[mask] + 1.0) * 0.5
    
    # -Z face (뒤)
    mask = (vec_z < 0) & (abs_z >= abs_x) & (abs_z >= abs_y)
    face_idx[mask] = 5
    u[mask] = (-vec_x[mask] / abs_z[mask] + 1.0) * 0.5
    v[mask] = (-vec_y[mask] / abs_z[mask] + 1.0) * 0.5
    
    return face_idx, u, v


def create_fisheye_mapping(height, width, fov=117.0, device='cuda'):
    """개선된 fisheye 매핑"""
    cx = width / 2.0
    cy = height / 2.0
    
    fov_rad = math.radians(fov)
    radius = min(cx, cy)
    
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # 정규화된 좌표
    dx = (x - cx) / radius
    dy = (y - cy) / radius
    r = torch.sqrt(dx**2 + dy**2)
    
    valid_mask = r <= 1.0
    
    # Equidistant projection
    theta = r * (fov_rad / 2.0)
    phi = torch.atan2(dy, dx)
    
    # 3D ray direction
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    sin_phi = torch.sin(phi)
    cos_phi = torch.cos(phi)
    
    vec_x = sin_theta * cos_phi
    vec_y = sin_theta * sin_phi
    vec_z = cos_theta
    
    # Cubemap face 결정 (수정된 함수 사용)
    face_idx, u, v = determine_cubemap_face(vec_x, vec_y, vec_z)
    
    return face_idx, u, v, valid_mask


def sample_cubemap(cubemap_faces, face_idx, u, v, valid_mask):
    """개선된 bilinear sampling"""
    device = face_idx.device
    C = cubemap_faces[0].shape[0]
    H, W = face_idx.shape
    face_h, face_w = cubemap_faces[0].shape[1:3]
    
    output = torch.zeros(C, H, W, device=device)
    
    for i in range(6):
        mask = (face_idx == i) & valid_mask
        if mask.sum() == 0:
            continue
        
        u_pixel = u[mask] * (face_w - 1)
        v_pixel = v[mask] * (face_h - 1)
        
        u0 = torch.floor(u_pixel).long().clamp(0, face_w - 1)
        u1 = (u0 + 1).clamp(0, face_w - 1)
        v0 = torch.floor(v_pixel).long().clamp(0, face_h - 1)
        v1 = (v0 + 1).clamp(0, face_h - 1)
        
        wu = (u_pixel - u0.float()).clamp(0, 1).unsqueeze(0)
        wv = (v_pixel - v0.float()).clamp(0, 1).unsqueeze(0)
        
        face = cubemap_faces[i]
        
        p00 = face[:, v0, u0]
        p01 = face[:, v0, u1]
        p10 = face[:, v1, u0]
        p11 = face[:, v1, u1]
        
        sampled = ((1 - wu) * (1 - wv) * p00 +
                  wu * (1 - wv) * p01 +
                  (1 - wu) * wv * p10 +
                  wu * wv * p11)
        
        output[:, mask] = sampled
    
    return output


def cubemap_to_fisheye(cubemap_faces, height, width, fov=117.0, mapping_cache=None):
    """
    Main function: Cubemap을 Fisheye 이미지로 변환
    
    Args:
        cubemap_faces (list of torch.Tensor): 6개의 cubemap face [C, H, W]
            Order: [+X, -X, +Y, -Y, +Z, -Z]
        height (int): 출력 fisheye 이미지 높이
        width (int): 출력 fisheye 이미지 너비
        fov (float): Field of view in degrees
        mapping_cache (dict, optional): 매핑 캐시 (속도 향상)
        
    Returns:
        torch.Tensor: Fisheye 이미지 [C, H, W]
    """
    device = cubemap_faces[0].device
    
    # 매핑이 캐시되어 있지 않으면 생성
    if mapping_cache is None:
        face_idx, u, v, valid_mask = create_fisheye_mapping(height, width, fov, device)
    else:
        face_idx = mapping_cache['face_idx']
        u = mapping_cache['u']
        v = mapping_cache['v']
        valid_mask = mapping_cache['valid_mask']
    
    # Cubemap에서 샘플링
    fisheye_image = sample_cubemap(cubemap_faces, face_idx, u, v, valid_mask)
    
    return fisheye_image


def create_mapping_cache(height, width, fov=117.0, device='cuda'):
    """
    매핑을 미리 계산해서 캐시로 저장 (학습 시 속도 향상)
    
    Args:
        height (int): Fisheye 이미지 높이
        width (int): Fisheye 이미지 너비
        fov (float): Field of view in degrees
        device (str): 'cuda' or 'cpu'
        
    Returns:
        dict: 매핑 캐시
    """
    face_idx, u, v, valid_mask = create_fisheye_mapping(height, width, fov, device)
    
    return {
        'face_idx': face_idx,
        'u': u,
        'v': v,
        'valid_mask': valid_mask
    }


# 테스트 코드
if __name__ == "__main__":
    print("Testing cubemap to fisheye conversion...")
    
    # 테스트용 cubemap 생성 (각 face를 다른 색으로)
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
        face = torch.ones(3, face_size, face_size, device=device)
        for c in range(3):
            face[c] *= color[c]
        cubemap_faces.append(face)
    
    # Fisheye 변환
    print("Converting cubemap to fisheye...")
    fisheye_height = 1024
    fisheye_width = 1024
    
    # 매핑 캐시 생성
    cache = create_mapping_cache(fisheye_height, fisheye_width, fov=117.0, device=device)
    
    # 변환
    fisheye = cubemap_to_fisheye(cubemap_faces, fisheye_height, fisheye_width, mapping_cache=cache)
    
    print(f"Output shape: {fisheye.shape}")
    print("Conversion successful!")
    
    # 이미지 저장 (optional)
    try:
        from torchvision.utils import save_image
        save_image(fisheye, 'test_fisheye.png')
        print("Saved test image to test_fisheye.png")
    except:
        print("Could not save image (torchvision not available)")