"""
Cubemap 회전 행렬 검증 스크립트
================================
각 cubemap face를 고유한 색상으로 칠한 뒤 fisheye로 변환하여
cameras.py의 face rotation과 cubemap_to_fisheye.py의 매핑이 일치하는지 확인

사용법:
    python verify_cubemap_mapping.py

출력:
    verify_output/
    ├── 01_expected_layout.png        # 이론적으로 예상되는 배치
    ├── 02_mapping_only.png           # cubemap_to_fisheye 매핑만 테스트 (회전 무관)
    ├── 03_camera_rotation_test.png   # FisheyeCamera 회전 적용 후 테스트
    ├── 04_face_index_map.png         # 각 fisheye 픽셀이 어느 face에서 오는지
    ├── 05_uv_map.png                 # UV 좌표 시각화
    └── report.txt                    # 각 face의 픽셀 수, 비율 등 통계
"""

import torch
import numpy as np
import math
import os
import sys

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_colored_faces(face_size=512, device='cuda'):
    """각 face를 고유 색상으로 생성"""
    colors = {
        0: ([1.0, 0.0, 0.0], '+X (Red)'),
        1: ([0.0, 1.0, 0.0], '-X (Green)'),
        2: ([0.0, 0.0, 1.0], '+Y (Blue)'),
        3: ([1.0, 1.0, 0.0], '-Y (Yellow)'),
        4: ([1.0, 0.0, 1.0], '+Z (Magenta)'),
        5: ([0.0, 1.0, 1.0], '-Z (Cyan)'),
    }
    
    faces = []
    for i in range(6):
        color, name = colors[i]
        face = torch.zeros(3, face_size, face_size, device=device)
        for c in range(3):
            face[c] = color[c]
        
        # 각 face에 방향 표시용 그래디언트 추가
        # 왼→오: 밝아짐, 위→아: 밝아짐
        grad_h = torch.linspace(0.3, 1.0, face_size, device=device).view(1, -1, 1).expand(3, face_size, face_size)
        grad_w = torch.linspace(0.3, 1.0, face_size, device=device).view(1, 1, -1).expand(3, face_size, face_size)
        
        face = face * grad_h * grad_w
        
        # 중앙에 십자 표시 (orientation 확인용)
        center = face_size // 2
        thickness = max(2, face_size // 100)
        # 가로 줄 (흰색)
        face[:, center-thickness:center+thickness, :] = 1.0
        # 세로 줄 (흰색)
        face[:, :, center-thickness:center+thickness] = 1.0
        # 왼쪽 위 사분면에 점 표시 (orientation marker)
        quarter = face_size // 4
        dot_size = max(4, face_size // 50)
        face[:, quarter-dot_size:quarter+dot_size, quarter-dot_size:quarter+dot_size] = 1.0
        
        faces.append(face)
    
    return faces, colors


def test_mapping_only(device='cuda'):
    """
    테스트 1: cubemap_to_fisheye 매핑만 테스트 (카메라 회전 무관)
    
    단색 cubemap face를 직접 매핑하여 fisheye 이미지가 올바른지 확인
    예상: 정면(+Z, Magenta)이 중앙에, 주변부에 다른 색들이 배치
    """
    from utils.cubemap_to_fisheye import cubemap_to_fisheye, create_mapping_cache
    
    face_size = 512
    fisheye_size = 1024
    fov = 117.0
    
    faces, colors = create_colored_faces(face_size, device)
    cache = create_mapping_cache(fisheye_size, fisheye_size, fov=fov, device=device)
    
    fisheye = cubemap_to_fisheye(faces, fisheye_size, fisheye_size, fov=fov, mapping_cache=cache)
    
    return fisheye, cache, colors


def test_face_index_map(device='cuda'):
    """
    테스트 2: 각 fisheye 픽셀이 어느 face에서 오는지 시각화
    """
    from utils.cubemap_to_fisheye import create_mapping_cache
    
    fisheye_size = 1024
    fov = 117.0
    
    cache = create_mapping_cache(fisheye_size, fisheye_size, fov=fov, device=device)
    
    face_idx = cache['face_idx']
    valid_mask = cache['valid_mask']
    
    # Face별 색상
    face_colors = torch.tensor([
        [1.0, 0.0, 0.0],  # 0: +X Red
        [0.0, 1.0, 0.0],  # 1: -X Green
        [0.0, 0.0, 1.0],  # 2: +Y Blue
        [1.0, 1.0, 0.0],  # 3: -Y Yellow
        [1.0, 0.0, 1.0],  # 4: +Z Magenta
        [0.0, 1.0, 1.0],  # 5: -Z Cyan
    ], device=device)
    
    vis = torch.zeros(3, fisheye_size, fisheye_size, device=device)
    for i in range(6):
        mask = (face_idx == i) & valid_mask
        for c in range(3):
            vis[c, mask] = face_colors[i, c]
    
    # 통계
    stats = {}
    total_valid = valid_mask.sum().item()
    for i in range(6):
        count = ((face_idx == i) & valid_mask).sum().item()
        stats[i] = {
            'count': count,
            'percentage': count / total_valid * 100 if total_valid > 0 else 0
        }
    
    return vis, stats


def test_uv_map(device='cuda'):
    """
    테스트 3: UV 좌표를 시각화
    U → Red 채널, V → Green 채널
    """
    from utils.cubemap_to_fisheye import create_mapping_cache
    
    fisheye_size = 1024
    fov = 117.0
    
    cache = create_mapping_cache(fisheye_size, fisheye_size, fov=fov, device=device)
    
    u = cache['u']
    v = cache['v']
    valid_mask = cache['valid_mask']
    
    vis = torch.zeros(3, fisheye_size, fisheye_size, device=device)
    vis[0, valid_mask] = u[valid_mask]  # R = U
    vis[1, valid_mask] = v[valid_mask]  # G = V
    vis[2, valid_mask] = 0.3            # B = constant (visibility)
    
    return vis


def test_camera_rotation_consistency(device='cuda'):
    """
    테스트 4: FisheyeCamera의 cubemap 회전과 매핑의 일관성 검증
    
    Identity rotation (R=I, T=0)인 FisheyeCamera를 만들고,
    각 cubemap face camera의 world_view_transform이
    매핑의 face 방향과 일치하는지 확인
    """
    from scene.cameras import FisheyeCamera
    
    R_identity = np.eye(3)
    T_zero = np.array([0.0, 0.0, 0.0])
    
    dummy_image = np.zeros((512, 512, 3), dtype=np.float32)
    
    cam = FisheyeCamera(
        colmap_id=0,
        R=R_identity,
        T=T_zero,
        fov=117.0,
        fisheye_params={},
        image=dummy_image,
        gt_alpha_mask=None,
        image_name="test",
        uid=0,
        data_device="cpu"
    )
    
    cubemap_cams = cam.cubemap_cameras
    face_names = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
    
    results = []
    for i, (ccam, name) in enumerate(zip(cubemap_cams, face_names)):
        # world_view_transform은 getWorld2View2(R,T).transpose(0,1)로 저장됨
        # getWorld2View2는 R.T를 사용하므로:
        #   wvt = [[R.T | t], [0|1]].T = [[R | 0], [t.T | 1]]  (열 기준)
        # 
        # cam2world rotation = wvt[:3,:3] = R
        # 3DGS에서 카메라는 +Z 방향을 렌더링함
        # render direction (world) = R @ [0,0,1] = R의 3번째 열 = wvt[:3, 2]
        
        wvt = ccam.world_view_transform.cpu().numpy()
        
        # 3DGS: 렌더 방향 = cam_z in world = col2 of cam2world rotation
        look_dir = wvt[:3, 2]
        
        center = ccam.camera_center.cpu().numpy()
        
        results.append({
            'face': i,
            'name': name,
            'R': ccam.R,
            'look_direction': look_dir,
            'camera_center': center,
            'FoVx': ccam.FoVx,
            'FoVy': ccam.FoVy,
        })
    
    return results


def create_expected_layout_image(fisheye_size=1024, device='cuda'):
    """
    이론적으로 예상되는 face 배치를 그림으로 생성
    
    Fisheye 이미지에서:
    - 중앙: +Z (front, 앞)
    - 오른쪽: +X (right)  
    - 왼쪽: -X (left)
    - 위: -Y (up, 카메라 좌표계에서 -Y가 위)
    - 아래: +Y (down)
    - 가장자리: -Z (back, 뒤)
    """
    import math
    
    fov = 117.0
    fov_rad = math.radians(fov)
    
    cx = fisheye_size / 2.0
    cy = fisheye_size / 2.0
    radius = min(cx, cy)
    
    vis = torch.zeros(3, fisheye_size, fisheye_size, device=device)
    
    # 각 방향에 텍스트 대신 색상 원으로 표시
    directions = [
        # (angle_deg, distance_ratio, color, label)
        (0, 0.0, [1.0, 0.0, 1.0], '+Z center'),     # 중앙: Magenta
        (0, 0.5, [1.0, 0.0, 0.0], '+X right'),       # 오른쪽: Red
        (180, 0.5, [0.0, 1.0, 0.0], '-X left'),      # 왼쪽: Green
        (90, 0.5, [0.0, 0.0, 1.0], '+Y down'),       # 아래: Blue (카메라 +Y = 아래, 이미지 y축 아래)
        (270, 0.5, [1.0, 1.0, 0.0], '-Y up'),        # 위: Yellow (카메라 -Y = 위, 이미지 y축 위)
    ]
    
    for angle_deg, dist_ratio, color, label in directions:
        angle_rad = math.radians(angle_deg)
        px = int(cx + dist_ratio * radius * math.cos(angle_rad))
        py = int(cy + dist_ratio * radius * math.sin(angle_rad))
        
        dot_size = fisheye_size // 20
        y_start = max(0, py - dot_size)
        y_end = min(fisheye_size, py + dot_size)
        x_start = max(0, px - dot_size)
        x_end = min(fisheye_size, px + dot_size)
        
        for c in range(3):
            vis[c, y_start:y_end, x_start:x_end] = color[c]
    
    # 원형 경계 표시
    y, x = torch.meshgrid(
        torch.arange(fisheye_size, device=device, dtype=torch.float32),
        torch.arange(fisheye_size, device=device, dtype=torch.float32),
        indexing='ij'
    )
    r = torch.sqrt((x - cx)**2 + (y - cy)**2) / radius
    circle_mask = (r > 0.95) & (r < 1.0)
    vis[:, circle_mask] = 0.5
    
    return vis


def run_all_tests(output_dir="verify_output"):
    """모든 테스트 실행"""
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Output: {output_dir}/")
    print("=" * 60)
    
    from torchvision.utils import save_image
    
    report_lines = []
    report_lines.append("Cubemap Mapping Verification Report")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # ============================================================
    # 테스트 1: 예상 배치도
    # ============================================================
    print("\n[1/5] Creating expected layout...")
    expected = create_expected_layout_image(device=device)
    save_image(expected, os.path.join(output_dir, "01_expected_layout.png"))
    report_lines.append("Test 1: Expected Layout")
    report_lines.append("  중앙: +Z (Magenta)")
    report_lines.append("  오른쪽: +X (Red)")
    report_lines.append("  왼쪽: -X (Green)")
    report_lines.append("  위: -Y (Yellow, 카메라 좌표계)")
    report_lines.append("  아래: +Y (Blue, 카메라 좌표계)")
    report_lines.append("")
    
    # ============================================================
    # 테스트 2: 매핑만 테스트
    # ============================================================
    print("[2/5] Testing cubemap_to_fisheye mapping...")
    fisheye, cache, colors = test_mapping_only(device)
    save_image(fisheye, os.path.join(output_dir, "02_mapping_only.png"))
    
    report_lines.append("Test 2: Mapping Only (cubemap_to_fisheye)")
    report_lines.append("  Color legend:")
    for i, (color, name) in colors.items():
        report_lines.append(f"    Face {i}: {name}")
    report_lines.append("  → 02_mapping_only.png과 01_expected_layout.png을 비교하세요")
    report_lines.append("  → 중앙이 Magenta(+Z)이면 매핑 정상")
    report_lines.append("")
    
    # ============================================================
    # 테스트 3: Face index map
    # ============================================================
    print("[3/5] Creating face index map...")
    face_map, stats = test_face_index_map(device)
    save_image(face_map, os.path.join(output_dir, "03_face_index_map.png"))
    
    report_lines.append("Test 3: Face Index Map")
    report_lines.append("  Face pixel distribution:")
    for i, s in stats.items():
        name = ['+X', '-X', '+Y', '-Y', '+Z', '-Z'][i]
        report_lines.append(f"    Face {i} ({name}): {s['count']:6d} pixels ({s['percentage']:5.2f}%)")
    report_lines.append("")
    
    # ============================================================
    # 테스트 4: UV map
    # ============================================================
    print("[4/5] Creating UV map...")
    uv_map = test_uv_map(device)
    save_image(uv_map, os.path.join(output_dir, "04_uv_map.png"))
    
    report_lines.append("Test 4: UV Map")
    report_lines.append("  R channel = U coordinate, G channel = V coordinate")
    report_lines.append("  → 각 face 내에서 부드러운 그래디언트가 보여야 정상")
    report_lines.append("")
    
    # ============================================================
    # 테스트 5: 카메라 회전 일관성
    # ============================================================
    print("[5/5] Testing camera rotation consistency...")
    rotation_results = test_camera_rotation_consistency(device)
    
    report_lines.append("Test 5: Camera Rotation Consistency")
    report_lines.append("  FisheyeCamera(R=I, T=0)의 cubemap face cameras:")
    report_lines.append("")
    
    expected_dirs = {
        0: np.array([1, 0, 0]),   # +X: 오른쪽
        1: np.array([-1, 0, 0]),  # -X: 왼쪽
        2: np.array([0, 1, 0]),   # +Y: 아래 (카메라 좌표계)
        3: np.array([0, -1, 0]),  # -Y: 위
        4: np.array([0, 0, 1]),   # +Z: 앞
        5: np.array([0, 0, -1]),  # -Z: 뒤
    }
    
    all_match = True
    for r in rotation_results:
        i = r['face']
        look = r['look_direction']
        expected = expected_dirs[i]
        
        # 방향 일치 확인 (cosine similarity)
        cos_sim = np.dot(look, expected) / (np.linalg.norm(look) * np.linalg.norm(expected) + 1e-8)
        match = cos_sim > 0.9
        
        status = "✓ MATCH" if match else "✗ MISMATCH"
        if not match:
            all_match = False
        
        line = f"    Face {i} ({r['name']:3s}): look={look.round(3)}, expected={expected}, cos_sim={cos_sim:.3f} {status}"
        report_lines.append(line)
        print(f"  {line}")
    
    report_lines.append("")
    if all_match:
        report_lines.append("  ✓ 모든 face의 회전이 예상과 일치합니다!")
    else:
        report_lines.append("  ✗ 일부 face의 회전이 불일치합니다. cameras.py 수정 필요!")
    
    # ============================================================
    # 리포트 저장
    # ============================================================
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print("\n" + "=" * 60)
    print(f"All tests complete. Results in {output_dir}/")
    print(f"Report: {report_path}")
    if all_match:
        print("✓ Camera rotations match the mapping!")
    else:
        print("✗ Camera rotation MISMATCH detected — fix needed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()