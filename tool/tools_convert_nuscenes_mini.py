import os
import shutil
import argparse
import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion

import sys
sys.path.append("tool")
import tensor

""" 用法：

python tool/tools_convert_nuscenes_mini.py \
  --dataroot /home/goghy/Lidar_AI_Solution/CUDA-BEVFusion/nuscenesmini \
  --version v1.0-mini \
  --out nuscenes-mini-frames \
  --max-samples 20 \
  --scene-index 0

"""



CAMERAS = [
    ("CAM_FRONT", "0-FRONT.jpg"),
    ("CAM_FRONT_RIGHT", "1-FRONT_RIGHT.jpg"),
    ("CAM_FRONT_LEFT", "2-FRONT_LEFT.jpg"),
    ("CAM_BACK", "3-BACK.jpg"),
    ("CAM_BACK_LEFT", "4-BACK_LEFT.jpg"),
    ("CAM_BACK_RIGHT", "5-BACK_RIGHT.jpg"),
]


def transform_matrix(translation, rotation, inverse=False):
    """
    Create a 4x4 transform matrix from nuScenes translation and quaternion rotation.
    """
    tm = np.eye(4, dtype=np.float32)
    q = Quaternion(rotation)
    tm[:3, :3] = q.rotation_matrix.astype(np.float32)
    tm[:3, 3] = np.array(translation, dtype=np.float32)

    if inverse:
        tm_inv = np.eye(4, dtype=np.float32)
        r = tm[:3, :3]
        t = tm[:3, 3]
        tm_inv[:3, :3] = r.T
        tm_inv[:3, 3] = -r.T @ t
        return tm_inv

    return tm


def get_sensor_to_global(nusc, sample_data_token):
    """
    nuScenes chain:
      sensor -> ego -> global
    """
    sd = nusc.get("sample_data", sample_data_token)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ego = nusc.get("ego_pose", sd["ego_pose_token"])

    sensor_to_ego = transform_matrix(cs["translation"], cs["rotation"], inverse=False)
    ego_to_global = transform_matrix(ego["translation"], ego["rotation"], inverse=False)

    return ego_to_global @ sensor_to_ego


def pad_intrinsic_4x4(intrinsic_3x3):
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = np.array(intrinsic_3x3, dtype=np.float32)
    return mat


def convert_one_sample(nusc, sample, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_sd = nusc.get("sample_data", lidar_token)
    lidar_path = nusc.get_sample_data_path(lidar_token)

    # points: nuScenes lidar is [x, y, z, intensity] originally.
    # CUDA-BEVFusion expects 5 features. The 5th is set to 0 here.
    pc = LidarPointCloud.from_file(lidar_path)
    points_4 = pc.points.T.astype(np.float16)  # [N, 4]

    points_5 = np.zeros((points_4.shape[0], 5), dtype=np.float16)
    points_5[:, :4] = points_4[:, :4]

    tensor.save(points_5, os.path.join(out_dir, "points.tensor"))


    lidar_to_global = get_sensor_to_global(nusc, lidar_token)
    global_to_lidar = np.linalg.inv(lidar_to_global).astype(np.float32)

    camera2lidar_list = []
    camera_intrinsics_list = []
    lidar2image_list = []
    img_aug_matrix_list = []

    for cam_name, dst_name in CAMERAS:
        cam_token = sample["data"][cam_name]
        cam_sd = nusc.get("sample_data", cam_token)
        cam_path = nusc.get_sample_data_path(cam_token)

        # Copy original image. CUDA-BEVFusion will resize/normalize internally.
        shutil.copyfile(cam_path, os.path.join(out_dir, dst_name))

        cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])

        camera_to_global = get_sensor_to_global(nusc, cam_token)
        global_to_camera = np.linalg.inv(camera_to_global).astype(np.float32)

        camera_to_lidar = global_to_lidar @ camera_to_global
        lidar_to_camera = global_to_camera @ lidar_to_global

        intrinsic_4x4 = pad_intrinsic_4x4(cam_cs["camera_intrinsic"])

        lidar_to_image = intrinsic_4x4 @ lidar_to_camera

        # In the official example, these matrices are stored as [1, 6, 4, 4].
        camera2lidar_list.append(camera_to_lidar.astype(np.float32))
        camera_intrinsics_list.append(intrinsic_4x4.astype(np.float32))
        lidar2image_list.append(lidar_to_image.astype(np.float32))

        # No extra image augmentation here.
        # Official model uses 256x704 input internally; main.cpp/pybev handles preprocessing.
        img_aug_matrix_list.append(np.eye(4, dtype=np.float32))

    camera2lidar = np.stack(camera2lidar_list, axis=0)[None].astype(np.float32)
    camera_intrinsics = np.stack(camera_intrinsics_list, axis=0)[None].astype(np.float32)
    lidar2image = np.stack(lidar2image_list, axis=0)[None].astype(np.float32)
    img_aug_matrix = np.stack(img_aug_matrix_list, axis=0)[None].astype(np.float32)

    tensor.save(camera2lidar, os.path.join(out_dir, "camera2lidar.tensor"))
    tensor.save(camera_intrinsics, os.path.join(out_dir, "camera_intrinsics.tensor"))
    tensor.save(lidar2image, os.path.join(out_dir, "lidar2image.tensor"))
    tensor.save(img_aug_matrix, os.path.join(out_dir, "img_aug_matrix.tensor"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default=os.path.expanduser("~/nuscenesmini"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--out", default="nuscenes-mini-frames")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--scene-index", type=int, default=0)
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    scene = nusc.scene[args.scene_index]
    sample_token = scene["first_sample_token"]

    print(f"Converting scene: {scene['name']}")
    print(f"Output dir: {args.out}")

    count = 0
    while sample_token and count < args.max_samples:
        sample = nusc.get("sample", sample_token)
        frame_dir = os.path.join(args.out, f"frame_{count:06d}")

        print(f"[{count}] {sample['token']} -> {frame_dir}")
        convert_one_sample(nusc, sample, frame_dir)

        sample_token = sample["next"]
        count += 1

    print(f"Done. Converted {count} frames.")


if __name__ == "__main__":
    main()
