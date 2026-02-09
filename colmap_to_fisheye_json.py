"""
COLMAP 데이터를 fisheye_cameras.json으로 변환
- OPENCV_FISHEYE 모델의 파라미터(fx, fy, cx, cy, k1-k4)를 저장
- FOV를 카메라 파라미터에서 자동 계산
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# COLMAP 로더 import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text


def compute_opencv_fisheye_fov(fx, fy, cx, cy, k1, k2, k3, k4, width, height):
    """
    OPENCV_FISHEYE 모델에서 실제 FOV를 계산
    
    OPENCV_FISHEYE (equidistant) model:
        theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9
        r_pixel = f * theta_d
    
    주의: 왜곡 계수에 따라 forward 함수가 비단조(non-monotonic)일 수 있음.
    이 경우 derivative가 0이 되는 지점(theta_max_valid)까지만 유효.
    FOV = min(2*theta_solved, 2*theta_max_valid)
    """
    f_avg = (fx + fy) / 2.0
    r_max = min(cx, cy, width - cx, height - cy)
    theta_d_target = r_max / f_avg
    
    def forward(theta):
        return theta + k1*theta**3 + k2*theta**5 + k3*theta**7 + k4*theta**9
    
    def forward_deriv(theta):
        return 1 + 3*k1*theta**2 + 5*k2*theta**4 + 7*k3*theta**6 + 9*k4*theta**8
    
    # Step 1: Find theta_max_valid where forward function is still monotonically increasing
    # (derivative > 0). Search in 0.1° steps.
    theta_max_valid = np.pi  # default
    for deg10 in range(1, 1800):  # 0.1° to 180°
        t = np.radians(deg10 / 10.0)
        if forward_deriv(t) <= 0:
            theta_max_valid = np.radians((deg10 - 1) / 10.0)
            break
    
    # Step 2: The max achievable theta_d in the valid range
    theta_d_at_max = forward(theta_max_valid)
    
    # Step 3: If target theta_d exceeds what the model can produce,
    # clamp to theta_max_valid (image edges beyond this are invalid anyway)
    if theta_d_target > theta_d_at_max:
        print(f"    Note: theta_d_target ({np.degrees(theta_d_target):.1f}°) > max valid theta_d ({np.degrees(theta_d_at_max):.1f}°)")
        print(f"    Clamping FOV to valid range: {2*np.degrees(theta_max_valid):.1f}°")
        theta = theta_max_valid
    else:
        # Step 4: Bisection search in [0, theta_max_valid] (guaranteed monotonic)
        lo, hi = 0.0, theta_max_valid
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if forward(mid) < theta_d_target:
                lo = mid
            else:
                hi = mid
            if hi - lo < 1e-12:
                break
        theta = (lo + hi) / 2.0
    
    fov_full = 2.0 * np.degrees(theta)
    return fov_full


def colmap_to_fisheye_json(colmap_path, images_folder, output_json="fisheye_cameras.json", fov_override=None, max_cameras=None):
    """
    COLMAP 데이터를 fisheye_cameras.json으로 변환
    
    Args:
        colmap_path: COLMAP sparse 폴더 경로 (예: data/sparse/0)
        images_folder: 이미지 폴더 경로 (예: data/images)
        output_json: 출력 JSON 파일 이름
        fov_override: FOV 수동 지정 (None이면 COLMAP 파라미터에서 자동 계산)
        max_cameras: 최대 카메라 수 (None이면 전체 사용, 숫자면 균등 샘플링)
    """
    
    # COLMAP 데이터 읽기
    try:
        print("Reading COLMAP binary files...")
        cameras_extrinsic_file = os.path.join(colmap_path, "images.bin")
        cameras_intrinsic_file = os.path.join(colmap_path, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        print("Binary files not found, trying text files...")
        cameras_extrinsic_file = os.path.join(colmap_path, "images.txt")
        cameras_intrinsic_file = os.path.join(colmap_path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    cameras = []
    
    # 카메라 키 목록 (정렬해서 재현성 보장)
    all_keys = sorted(cam_extrinsics.keys())
    
    # max_cameras가 지정되면 균등 샘플링
    if max_cameras is not None and max_cameras < len(all_keys):
        indices = np.linspace(0, len(all_keys) - 1, max_cameras, dtype=int)
        selected_keys = [all_keys[i] for i in indices]
        print(f"Sampling {max_cameras} cameras from {len(all_keys)} total (uniform)")
    else:
        selected_keys = all_keys
    
    print(f"Processing {len(selected_keys)} cameras...")
    
    # 카메라 intrinsics 정보 출력
    for cam_id, intr in cam_intrinsics.items():
        print(f"\n  Camera {cam_id}: model={intr.model}, {intr.width}x{intr.height}")
        print(f"    params={intr.params}")
        
        if intr.model == "OPENCV_FISHEYE":
            fx, fy, cx, cy = intr.params[0], intr.params[1], intr.params[2], intr.params[3]
            k1, k2, k3, k4 = intr.params[4], intr.params[5], intr.params[6], intr.params[7]
            
            if fov_override is None:
                auto_fov = compute_opencv_fisheye_fov(fx, fy, cx, cy, k1, k2, k3, k4, intr.width, intr.height)
                print(f"    Auto-calculated FOV: {auto_fov:.1f}°")
            else:
                print(f"    Using override FOV: {fov_override:.1f}°")
    
    for idx, key in enumerate(selected_keys):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        
        # Rotation matrix
        R = np.transpose(qvec2rotmat(extr.qvec))
        
        # Translation
        T = np.array(extr.tvec)
        
        # Image info
        image_name = extr.name
        image_path = os.path.join(images_folder, image_name)
        
        # 이미지가 없으면 basename으로도 시도
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, os.path.basename(image_name))
            image_name = os.path.basename(image_name)
        
        # Get image size
        if os.path.exists(image_path):
            img = Image.open(image_path)
            width, height = img.size
        else:
            print(f"\n⚠ Warning: Image not found: {image_path}")
            print(f"  COLMAP name: {extr.name}")
            print(f"  Available files (first 5): {os.listdir(images_folder)[:5]}")
            print(f"  Using COLMAP dimensions: {intr.width}x{intr.height}")
            width = intr.width
            height = intr.height
        
        # OPENCV_FISHEYE 파라미터 추출
        fisheye_params = {}
        fov = fov_override if fov_override is not None else 190.0  # fallback default
        
        if intr.model == "OPENCV_FISHEYE":
            fx, fy, cx_i, cy_i = intr.params[0], intr.params[1], intr.params[2], intr.params[3]
            k1, k2, k3, k4 = intr.params[4], intr.params[5], intr.params[6], intr.params[7]
            
            fisheye_params = {
                "distortion_model": "opencv_fisheye",
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx_i),
                "cy": float(cy_i),
                "k1": float(k1),
                "k2": float(k2),
                "k3": float(k3),
                "k4": float(k4),
                "colmap_width": int(intr.width),
                "colmap_height": int(intr.height),
            }
            
            if fov_override is None:
                fov = compute_opencv_fisheye_fov(fx, fy, cx_i, cy_i, k1, k2, k3, k4, intr.width, intr.height)
        else:
            fisheye_params = {
                "distortion_model": "equidistant"
            }
        
        # Camera dict
        cam_dict = {
            "image_name": image_name,
            "R": R.tolist(),
            "T": T.tolist(),
            "fov": round(fov, 2),
            "width": width,
            "height": height,
            "fisheye_params": fisheye_params
        }
        
        cameras.append(cam_dict)
        print(f"  [{idx+1}/{len(selected_keys)}] {image_name} (FOV={fov:.1f}°)")
    
    # JSON 생성
    output_data = {
        "cameras": cameras
    }
    
    # 저장
    output_path = os.path.join(os.path.dirname(colmap_path), "..", output_json)
    output_path = os.path.normpath(output_path)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"\n✅ Successfully created: {output_path}")
    print(f"   Total cameras: {len(cameras)}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert COLMAP to fisheye_cameras.json")
    parser.add_argument("--colmap", "-c", type=str, required=True, 
                        help="Path to COLMAP sparse folder (e.g., data/sparse/0)")
    parser.add_argument("--images", "-i", type=str, required=True,
                        help="Path to images folder (e.g., data/images)")
    parser.add_argument("--output", "-o", type=str, default="fisheye_cameras.json",
                        help="Output JSON filename")
    parser.add_argument("--fov", type=float, default=None,
                        help="Override fisheye FOV in degrees (auto-calculated from COLMAP params if not set)")
    parser.add_argument("--max_cameras", type=int, default=None,
                        help="Maximum number of cameras to include (uniformly sampled)")
    
    args = parser.parse_args()
    
    colmap_to_fisheye_json(args.colmap, args.images, args.output, args.fov, args.max_cameras)