"""
COLMAP 데이터를 fisheye_cameras.json으로 변환
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


def colmap_to_fisheye_json(colmap_path, images_folder, output_json="fisheye_cameras.json", fov=117.0):
    """
    COLMAP 데이터를 fisheye_cameras.json으로 변환
    
    Args:
        colmap_path: COLMAP sparse 폴더 경로 (예: data/sparse/0)
        images_folder: 이미지 폴더 경로 (예: data/images)
        output_json: 출력 JSON 파일 이름
        fov: Fisheye FOV (degrees)
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
        
        # Camera dict
        cam_dict = {
            "image_name": image_name,
            "R": R.tolist(),
            "T": T.tolist(),
            "fov": fov,
            "width": width,
            "height": height,
            "fisheye_params": {
                "distortion_model": "equidistant"
            }
        }
        
        cameras.append(cam_dict)
        print(f"  [{idx+1}/{len(cam_extrinsics)}] {image_name}")
    
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
    parser.add_argument("--fov", type=float, default=117.0,
                        help="Fisheye FOV in degrees (default: 180)")
    
    args = parser.parse_args()
    
    colmap_to_fisheye_json(args.colmap, args.images, args.output, args.fov)