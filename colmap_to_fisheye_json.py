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
    
    r_max = 이미지 원의 반지름 (principal point에서 가장 가까운 edge까지)
    theta_d_max = r_max / f
    theta_max = solve(theta_d = theta_d_max) via Newton's method
    FOV = 2 * theta_max
    """
    f_avg = (fx + fy) / 2.0
    r_max = min(cx, cy, width - cx, height - cy)
    theta_d_max = r_max / f_avg
    
    # Newton's method to solve: theta + k1*t^3 + k2*t^5 + k3*t^7 + k4*t^9 = theta_d_max
    theta = theta_d_max  # initial guess
    for _ in range(50):
        t3 = theta**3
        t5 = theta**5
        t7 = theta**7
        t9 = theta**9
        f_val = theta + k1*t3 + k2*t5 + k3*t7 + k4*t9 - theta_d_max
        f_deriv = 1 + 3*k1*theta**2 + 5*k2*theta**4 + 7*k3*theta**6 + 9*k4*theta**8
        if abs(f_deriv) < 1e-12:
            break
        theta = theta - f_val / f_deriv
    
    fov_full = 2.0 * np.degrees(theta)
    return fov_full


def colmap_to_fisheye_json(colmap_path, images_folder, output_json="fisheye_cameras.json", fov_override=None):
    """
    COLMAP 데이터를 fisheye_cameras.json으로 변환
    
    Args:
        colmap_path: COLMAP sparse 폴더 경로 (예: data/sparse/0)
        images_folder: 이미지 폴더 경로 (예: data/images)
        output_json: 출력 JSON 파일 이름
        fov_override: FOV 수동 지정 (None이면 COLMAP 파라미터에서 자동 계산)
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
    
    print(f"Processing {len(cam_extrinsics)} cameras...")
    
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
    
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        
        # Rotation matrix
        R = np.transpose(qvec2rotmat(extr.qvec))
        
        # Translation
        T = np.array(extr.tvec)
        
        # Image info
        image_name = extr.name
        image_path = os.path.join(images_folder, image_name)
        
        # Get image size
        if os.path.exists(image_path):
            img = Image.open(image_path)
            width, height = img.size
        else:
            print(f"Warning: Image not found: {image_path}, using COLMAP dimensions")
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
        print(f"  [{idx+1}/{len(cam_extrinsics)}] {image_name} (FOV={fov:.1f}°)")
    
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
    
    args = parser.parse_args()
    
    colmap_to_fisheye_json(args.colmap, args.images, args.output, args.fov)