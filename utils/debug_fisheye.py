"""
Fisheye 렌더링 디버깅 유틸리티
"""
import torch
import torchvision
import os
import numpy as np


def save_cubemap_debug(cubemap_faces, output_dir):
    """Cubemap face들을 개별적으로 저장 + 통합 이미지"""
    os.makedirs(output_dir, exist_ok=True)
    
    face_names = ['pos_x', 'neg_x', 'pos_y', 'neg_y', 'pos_z', 'neg_z']
    
    print(f"\n{'='*50}")
    print(f"Cubemap Faces → {output_dir}")
    print(f"{'='*50}")
    
    for i, (face, name) in enumerate(zip(cubemap_faces, face_names)):
        face_detached = face.detach()
        path = os.path.join(output_dir, f'cubemap_{i}_{name}.png')
        torchvision.utils.save_image(face_detached, path)
        
        print(f"  Face {i} ({name:6s}): shape={face.shape}, "
              f"min={face_detached.min():.3f}, max={face_detached.max():.3f}, "
              f"mean={face_detached.mean():.3f}")
    
    # 6개 face를 2x3 그리드로
    C, H, W = cubemap_faces[0].shape
    grid = torch.zeros(C, 2*H, 3*W, device=cubemap_faces[0].device)
    indices = [1, 4, 0, 3, 5, 2]  # left, front, right / down, back, up
    
    for idx_pos, face_idx in enumerate(indices):
        row = idx_pos // 3
        col = idx_pos % 3
        grid[:, row*H:(row+1)*H, col*W:(col+1)*W] = cubemap_faces[face_idx].detach()
    
    combined_path = os.path.join(output_dir, 'cubemap_combined.png')
    torchvision.utils.save_image(grid, combined_path)
    print(f"  ✓ Combined: cubemap_combined.png")
    print(f"{'='*50}\n")


def save_fisheye_comparison(rendered, gt, output_path):
    """렌더링 결과와 GT 비교 (좌: rendered, 우: GT)"""
    rendered_detached = rendered.detach().clamp(0, 1)
    gt_detached = gt.detach().clamp(0, 1)
    
    # 크기가 다를 경우 처리
    if rendered_detached.shape != gt_detached.shape:
        print(f"  ⚠ Shape mismatch: rendered={rendered_detached.shape}, gt={gt_detached.shape}")
        min_h = min(rendered_detached.shape[1], gt_detached.shape[1])
        min_w = min(rendered_detached.shape[2], gt_detached.shape[2])
        rendered_detached = rendered_detached[:, :min_h, :min_w]
        gt_detached = gt_detached[:, :min_h, :min_w]
    
    comparison = torch.cat([rendered_detached, gt_detached], dim=2)
    torchvision.utils.save_image(comparison, output_path)
    
    # PSNR 계산
    mse = ((rendered_detached - gt_detached) ** 2).mean()
    if mse > 0:
        psnr_val = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))
        print(f"  ✓ Comparison saved: {os.path.basename(output_path)} (PSNR: {psnr_val:.2f} dB)")
    else:
        print(f"  ✓ Comparison saved: {os.path.basename(output_path)}")


def visualize_mapping(face_idx, height, width, output_path):
    """Face index를 색상으로 시각화"""
    colors = torch.tensor([
        [1.0, 0.0, 0.0],  # 0: Red (+X)
        [0.0, 1.0, 0.0],  # 1: Green (-X)
        [0.0, 0.0, 1.0],  # 2: Blue (+Y)
        [1.0, 1.0, 0.0],  # 3: Yellow (-Y)
        [1.0, 0.0, 1.0],  # 4: Magenta (+Z)
        [0.0, 1.0, 1.0],  # 5: Cyan (-Z)
    ], dtype=torch.float32, device=face_idx.device)
    
    vis = torch.zeros(3, height, width, dtype=torch.float32, device=face_idx.device)
    
    for i in range(6):
        mask = face_idx == i
        if mask.sum() > 0:
            count = mask.sum().item()
            pct = count / (height * width) * 100
            for c in range(3):
                vis[c, mask] = colors[i, c]
            print(f"  Face {i}: {count:6d} pixels ({pct:5.2f}%)")
    
    torchvision.utils.save_image(vis, output_path)
    print(f"  ✓ Mapping visualization: {os.path.basename(output_path)}")


def verify_gradient_flow(render_pkg, loss):
    """Gradient flow가 정상적으로 작동하는지 확인"""
    print("\n" + "="*50)
    print("Gradient Flow Verification")
    print("="*50)
    
    # fisheye image가 grad를 갖고 있는지
    image = render_pkg["render"]
    print(f"  Fisheye image requires_grad: {image.requires_grad}")
    print(f"  Fisheye image grad_fn: {image.grad_fn}")
    
    # cubemap faces
    if "cubemap_faces" in render_pkg:
        for i, face in enumerate(render_pkg["cubemap_faces"]):
            print(f"  Cubemap face {i} requires_grad: {face.requires_grad}, grad_fn: {face.grad_fn}")
    
    # viewspace points
    if "viewspace_points_list" in render_pkg:
        for i, vp in enumerate(render_pkg["viewspace_points_list"]):
            has_grad = vp.grad is not None
            grad_norm = vp.grad.norm().item() if has_grad else 0
            print(f"  Viewspace points {i}: has_grad={has_grad}, norm={grad_norm:.6f}")
    
    print("="*50 + "\n")