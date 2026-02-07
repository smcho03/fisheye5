#
# Fisheye Camera 지원 추가 함수들
# 기존 camera_utils.py에 추가할 내용
#
import numpy as np
from scene.cameras import Camera, FisheyeCamera


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    """
    CameraInfo 리스트에서 Camera 또는 FisheyeCamera 객체 생성
    """
    camera_list = []

    for id, c in enumerate(cam_infos):
        # Check if fisheye camera
        if hasattr(c, 'is_fisheye') and c.is_fisheye:
            # Create FisheyeCamera
            camera_list.append(loadFisheyeCam(args, id, c, resolution_scale))
        else:
            # Create regular Camera
            camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def loadCam(args, id, cam_info, resolution_scale):
    """
    일반 Pinhole Camera 로드
    """
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    depth_image = None
    if cam_info.depth_image is not None:
        depth_image = cv2.resize(cam_info.depth_image, resolution, interpolation=cv2.INTER_NEAREST)

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  depth_params=cam_info.depth_params,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp = args.train_test_exp, is_test_dataset = False, 
                  scene_scale=getattr(args, 'scene_scale', 1.0),
                  depth_image=depth_image)


def loadFisheyeCam(args, id, cam_info, resolution_scale):
    """
    Fisheye Camera 로드
    """
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    depth_image = None
    if cam_info.depth_image is not None:
        depth_image = cv2.resize(cam_info.depth_image, resolution, interpolation=cv2.INTER_NEAREST)

    # FOV는 FovY 또는 FovX 중 하나를 사용 (fisheye는 동일)
    fov_degrees = np.rad2deg(cam_info.FovY)
    
    return FisheyeCamera(
        colmap_id=cam_info.uid, 
        R=cam_info.R, 
        T=cam_info.T, 
        fov=fov_degrees,
        fisheye_params=cam_info.fisheye_params if hasattr(cam_info, 'fisheye_params') else {},
        image=gt_image, 
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name, 
        uid=id, 
        data_device=args.data_device,
        train_test_exp=args.train_test_exp, 
        is_test_dataset=False, 
        scene_scale=getattr(args, 'scene_scale', 1.0),
        depth_image=depth_image
    )


def camera_to_JSON(id, camera):
    # CameraInfo인지 Camera 객체인지 확인
    if hasattr(camera, 'image_width'):
        # Camera 객체
        width = camera.image_width
        height = camera.image_height
    else:
        # CameraInfo
        width = camera.width
        height = camera.height
    
    # 나머지 코드도 수정
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    
    # Fisheye 체크
    if hasattr(camera, 'is_fisheye') and camera.is_fisheye:
        camera_dict = {
            'id': id,
            'img_name': camera.image_name,
            'width': width,
            'height': height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'fov': camera.FovY if hasattr(camera, 'FovY') else 117.0,
            'is_fisheye': True
        }
    else:
        # Pinhole
        fovY = camera.FovY if hasattr(camera, 'FovY') else 0.0
        fovX = camera.FovX if hasattr(camera, 'FovX') else 0.0
        camera_dict = {
            'id': id,
            'img_name': camera.image_name,
            'width': width,
            'height': height,
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'fy': fov2focal(fovY, height),
            'fx': fov2focal(fovX, width)
        }
    
    return camera_dict


# 필요한 helper 함수들
def PILtoTorch(pil_image, resolution):
    """PIL Image를 Torch Tensor로 변환"""
    import torch
    from torchvision import transforms
    
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def fov2focal(fov, pixels):
    """FOV를 focal length로 변환"""
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal, pixels):
    """Focal length를 FOV로 변환"""
    return 2*np.arctan(pixels/(2*focal))


# Global warning flag
WARNED = False