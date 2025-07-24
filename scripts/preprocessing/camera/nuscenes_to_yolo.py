import os
import shutil
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
import numpy as np
from tqdm import tqdm   # Tambahkan ini

CAMERAS = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]

DATAROOT = '../../../data/raw/nuscenes'
IMAGE_OUT_DIR = '../../../data/processed/nuscenes/camera/images'
LABEL_OUT_DIR = '../../../data/processed/nuscenes/camera/labels'

CLASS_MAPPING = {
    'human.pedestrian.adult': 0,
    'human.pedestrian.child': 1,
    'human.pedestrian.wheelchair': 2,
    'human.pedestrian.stroller': 3,
    'human.pedestrian.personal_mobility': 4,
    'human.pedestrian.police_officer': 5,
    'human.pedestrian.construction_worker': 6,
    'animal': 7,
    'vehicle.car': 8,
    'vehicle.motorcycle': 9,
    'vehicle.bicycle': 10,
    'vehicle.bus.bendy': 11,
    'vehicle.bus.rigid': 12,
    'vehicle.truck': 13,
    'vehicle.construction': 14,
    'vehicle.emergency.ambulance': 15,
    'vehicle.emergency.police': 16,
    'vehicle.trailer': 17,
    'movable_object.barrier': 18,
    'movable_object.trafficcone': 19,
    'movable_object.pushable_pullable': 20,
    'movable_object.debris': 21,
    'static_object.bicycle_rack': 22
}

IMG_WIDTH = 1600
IMG_HEIGHT = 900

def corners_to_yolo_bbox(corners_2d):
    xs = corners_2d[0]
    ys = corners_2d[1]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    x_center = (x_min + x_max) / 2.0 / IMG_WIDTH
    y_center = (y_min + y_max) / 2.0 / IMG_HEIGHT
    width = (x_max - x_min) / IMG_WIDTH
    height = (y_max - y_min) / IMG_HEIGHT
    return x_center, y_center, width, height

def is_yolo_bbox_valid(x_center, y_center, width, height):
    return (
        0.0 <= x_center <= 1.0 and
        0.0 <= y_center <= 1.0 and
        0.01 < width <= 1.0 and
        0.01 < height <= 1.0
    )

def main():
    os.makedirs(IMAGE_OUT_DIR, exist_ok=True)
    os.makedirs(LABEL_OUT_DIR, exist_ok=True)

    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=True)
    print(f"\n=== Proses {len(nusc.sample)} sample nuScenes ===\n")
    # Gunakan tqdm untuk progress bar pada sample
    for idx, sample in enumerate(tqdm(nusc.sample, desc="Processing samples", unit="sample")):
        sample_token = sample['token']
        sample_data = nusc.get('sample', sample_token)
        for cam in CAMERAS:
            camera_token = sample_data['data'][cam]
            camera_data = nusc.get('sample_data', camera_token)
            calibrated_sensor = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
            camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
            sensor_translation = np.array(calibrated_sensor['translation'])
            sensor_rotation = Quaternion(calibrated_sensor['rotation'])

            ego_pose = nusc.get('ego_pose', camera_data['ego_pose_token'])
            ego_translation = np.array(ego_pose['translation'])
            ego_rotation = Quaternion(ego_pose['rotation'])

            ann_tokens = sample_data['anns']
            label_lines = []

            for ann_token in ann_tokens:
                ann = nusc.get('sample_annotation', ann_token)
                category = ann['category_name']
                class_id = CLASS_MAPPING.get(category, -1)
                if class_id == -1:
                    continue
                box = Box(
                    center=ann['translation'],
                    size=ann['size'],
                    orientation=Quaternion(ann['rotation']),
                    name=category
                )

                box.translate(-ego_translation)
                box.rotate(ego_rotation.inverse)
                box.translate(-sensor_translation)
                box.rotate(sensor_rotation.inverse)

                corners_3d = box.corners()
                corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)

                x_center, y_center, width, height = corners_to_yolo_bbox(corners_2d[:2])
                valid = is_yolo_bbox_valid(x_center, y_center, width, height)
                if valid:
                    label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Copy image ke folder output dan buat label dengan nama yg seragam
            image_src_path = os.path.join(DATAROOT, camera_data['filename'])
            image_name = os.path.basename(camera_data['filename'])  # misal: 'abc123.jpg'
            image_dst_path = os.path.join(IMAGE_OUT_DIR, image_name)
            label_dst_path = os.path.join(LABEL_OUT_DIR, image_name.replace('.jpg', '.txt'))

            # Copy image
            if not os.path.exists(image_dst_path):
                shutil.copy2(image_src_path, image_dst_path)
            # Tulis label
            with open(label_dst_path, "w") as f:
                for line in label_lines:
                    f.write(line + "\n")

    print(f"\n=== Proses selesai. Images di {IMAGE_OUT_DIR} dan label YOLO di {LABEL_OUT_DIR} ===\n")

if __name__ == "__main__":
    main()