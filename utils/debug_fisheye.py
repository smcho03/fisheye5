import torch
import torchvision
import os
import matplotlib.pyplot as plt
import numpy as np

def save_cubemap_debug(cubemap_faces, output_dir):
    """Cubemap face들을 개별적으로 저장 + 통합 이미지"""
    os.makedirs(output_dir, exist_ok=True)
    
    face_names = ['right', 'left', 'up', 'down', 'front', 'back']
    
    print(f"\n{'='*60}")
    print(f"Saving Cubemap Faces to {output_dir}")
    print(f"{'='*60}")
    
    for i, (face, name) in enumerate(zip(cubemap_faces, face_names)):
        # 개별 저장
        path = os.path.join(output_dir, f'cubemap_{i}_{name}.png')
        torchvision.utils.save_image(face, path)
        
        # 통계 출력
        print(f"  Face {i} ({name:6s}): shape={face.shape}, "
              f"min={face.min():.3f}, max={face.max():.3f}, "
              f"mean={face.mean():.3f}")
    
    # 🔴 모든 face를 한 이미지로 합치기 (2x3 그리드)
    combined = create_cubemap_grid(cubemap_faces)
    combined_path = os.path.join(output_dir, 'cubemap_combined.png')
    torchvision.utils.save_image(combined, combined_path)
    print(f"\n  ✓ Saved combined cubemap to: cubemap_combined.png")
    print(f"{'='*60}\n")


def create_cubemap_grid(cubemap_faces):
    """6개 face를 2x3 그리드로 배치"""
    # [right, left, up, down, front, back]
    # 배치:
    # [left] [front] [right]
    # [down] [back]  [up]
    
    indices = [1, 4, 0, 3, 5, 2]  # 재배치 순서
    
    C, H, W = cubemap_faces[0].shape
    
    # 2x3 그리드 생성
    grid = torch.zeros(C, 2*H, 3*W, device=cubemap_faces[0].device)
    
    for idx, face_idx in enumerate(indices):
        row = idx // 3
        col = idx % 3
        grid[:, row*H:(row+1)*H, col*W:(col+1)*W] = cubemap_faces[face_idx]
    
    return grid


def visualize_mapping(face_idx, height, width, output_path):
    """Face index를 색상으로 시각화 - 개선 버전"""
    
    print(f"\n{'='*60}")
    print(f"Creating Mapping Visualization")
    print(f"{'='*60}")
    print(f"  Output size: {height}x{width}")
    print(f"  face_idx shape: {face_idx.shape}")
    print(f"  face_idx dtype: {face_idx.dtype}")
    print(f"  face_idx device: {face_idx.device}")
    
    # 각 face의 픽셀 수 출력
    for i in range(6):
        count = (face_idx == i).sum().item()
        percentage = count / (height * width) * 100
        print(f"  Face {i}: {count:6d} pixels ({percentage:5.2f}%)")
    
    # 🔴 더 선명한 색상
    colors = torch.tensor([
        [1.0, 0.0, 0.0],  # 0: Red (right)
        [0.0, 1.0, 0.0],  # 1: Green (left)
        [0.0, 0.0, 1.0],  # 2: Blue (up)
        [1.0, 1.0, 0.0],  # 3: Yellow (down)
        [1.0, 0.0, 1.0],  # 4: Magenta (front)
        [0.0, 1.0, 1.0],  # 5: Cyan (back)
    ], dtype=torch.float32, device=face_idx.device)
    
    vis = torch.zeros(3, height, width, dtype=torch.float32, device=face_idx.device)
    
    # 각 face별로 색칠
    for i in range(6):
        mask = face_idx == i
        if mask.sum() > 0:
            for c in range(3):
                vis[c, mask] = colors[i, c]
    
    # 🔴 matplotlib로 추가 시각화 (레전드 포함)
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 왼쪽: 색상 매핑
        img_np = vis.cpu().numpy().transpose(1, 2, 0)
        ax1.imshow(img_np)
        ax1.set_title('Fisheye Mapping (Face Colors)', fontsize=16)
        ax1.axis('off')
        
        # 오른쪽: face_idx 숫자로
        face_idx_np = face_idx.cpu().numpy()
        im = ax2.imshow(face_idx_np, cmap='tab10', vmin=0, vmax=5)
        ax2.set_title('Face Index Numbers', fontsize=16)
        ax2.axis('off')
        
        # 컬러바 추가
        cbar = plt.colorbar(im, ax=ax2, ticks=range(6))
        cbar.set_label('Face Index', fontsize=12)
        cbar.ax.set_yticklabels(['0:Right', '1:Left', '2:Up', 
                                  '3:Down', '4:Front', '5:Back'])
        
        # 저장
        plt_path = output_path.replace('.png', '_detailed.png')
        plt.tight_layout()
        plt.savefig(plt_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  ✓ Saved detailed visualization: {os.path.basename(plt_path)}")
        
    except Exception as e:
        print(f"  ⚠ Matplotlib visualization failed: {e}")
    
    # 기본 이미지도 저장
    torchvision.utils.save_image(vis, output_path)
    print(f"  ✓ Saved basic visualization: {os.path.basename(output_path)}")
    print(f"{'='*60}\n")


def save_fisheye_comparison(rendered, gt, output_path):
    """렌더링 결과와 GT 비교"""
    comparison = torch.cat([rendered, gt], dim=2)  # 좌우로 나란히
    torchvision.utils.save_image(comparison, output_path)
    print(f"  ✓ Saved comparison: {os.path.basename(output_path)}")