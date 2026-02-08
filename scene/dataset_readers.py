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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth_params: dict
    is_fisheye: bool = False
    fisheye_params: dict = None
    depth_image: np.array = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        if len(cam_centers) == 0:
            print("Warning: No cameras found!")
            return np.array([0., 0., 0.]), 1.0
        
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])
    
    if len(cam_centers) == 0:
        print("Warning: No camera centers collected!")
        return {"translate": np.array([0., 0., 0.]), "radius": 1.0}

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, depths_folder=None):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        depth_params = {}
        depth_image = None
        if depths_folder is not None:
            depth_image_path = os.path.join(depths_folder, os.path.basename(extr.name))
            if os.path.exists(depth_image_path):
                depth_image = np.array(Image.open(depth_image_path))
                
                depth_params_path = os.path.join(depths_folder, "depth_params.json")
                if os.path.exists(depth_params_path):
                    with open(depth_params_path, 'r') as f:
                        all_depth_params = json.load(f)
                        depth_params = all_depth_params.get(image_name, {})

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              depth_params=depth_params, depth_image=depth_image)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readFisheyeCameras(fisheye_json_path, images_folder, depths_folder=None):
    """
    Fisheye 카메라 정보를 JSON 파일에서 읽기
    
    fisheye_params에 OPENCV_FISHEYE 파라미터(fx, fy, cx, cy, k1-k4)가 있으면
    정확한 distortion model을 사용함
    """
    cam_infos = []
    
    with open(fisheye_json_path, 'r') as f:
        data = json.load(f)
    
    cameras = data.get("cameras", [])
    
    for idx, cam_data in enumerate(cameras):
        sys.stdout.write('\r')
        sys.stdout.write("Reading fisheye camera {}/{}".format(idx+1, len(cameras)))
        sys.stdout.flush()
        
        image_name = cam_data["image_name"]
        R = np.array(cam_data["R"])
        T = np.array(cam_data["T"])
        fov = cam_data.get("fov", 190.0)  # Changed default: 117 → 190
        width = cam_data["width"]
        height = cam_data["height"]
        fisheye_params = cam_data.get("fisheye_params", {})
        
        # Load image
        image_path = os.path.join(images_folder, image_name)
        if not os.path.exists(image_path):
            print(f"\nWarning: Image not found: {image_path}")
            continue
        
        image = Image.open(image_path)
        
        # Load depth if available
        depth_params = {}
        depth_image = None
        if depths_folder is not None:
            depth_image_path = os.path.join(depths_folder, image_name)
            if os.path.exists(depth_image_path):
                depth_image = np.array(Image.open(depth_image_path))
                
                depth_params_path = os.path.join(depths_folder, "depth_params.json")
                if os.path.exists(depth_params_path):
                    with open(depth_params_path, 'r') as f:
                        all_depth_params = json.load(f)
                        img_name_no_ext = os.path.splitext(image_name)[0]
                        depth_params = all_depth_params.get(img_name_no_ext, {})
        
        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=np.deg2rad(fov),
            FovX=np.deg2rad(fov),
            image=image,
            image_path=image_path,
            image_name=os.path.splitext(image_name)[0],
            width=width,
            height=height,
            depth_params=depth_params,
            is_fisheye=True,
            fisheye_params=fisheye_params,
            depth_image=depth_image
        )
        cam_infos.append(cam_info)
    
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, depths=None, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    
    depths_folder = None
    if depths is not None:
        depths_folder = os.path.join(path, depths)
    
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir),
                                           depths_folder=depths_folder)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readFisheyeSceneInfo(path, images, eval, fisheye_json="fisheye_cameras.json", depths=None, llffhold=8):
    """
    Fisheye 데이터셋 읽기
    """
    reading_dir = "images" if images == None else images
    images_folder = os.path.join(path, reading_dir)
    
    depths_folder = None
    if depths is not None:
        depths_folder = os.path.join(path, depths)
    
    fisheye_json_path = os.path.join(path, fisheye_json)
    if not os.path.exists(fisheye_json_path):
        raise FileNotFoundError(f"Fisheye camera info not found: {fisheye_json_path}")
    
    cam_infos = readFisheyeCameras(fisheye_json_path, images_folder, depths_folder)
    
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            try:
                xyz, rgb, _ = read_points3D_text(txt_path)
            except:
                print("Warning: No point cloud found, using random initialization")
                xyz = np.random.random((100, 3)) * 10 - 5
                rgb = np.random.random((100, 3))
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            depth_params={}))
            
    return cam_infos


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Fisheye": readFisheyeSceneInfo
}