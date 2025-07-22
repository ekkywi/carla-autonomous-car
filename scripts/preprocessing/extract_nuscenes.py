"""
Script untuk extract data mentah dari nuScenes ke folder lokal, 
dengan kamera otomatis di-resize ke 640x640 pixel agar konsisten dengan pipeline YOLO.

- Mengambil gambar kamera (resize 640x640), point cloud lidar, radar, dan annotation (JSON)
- Menyimpan ke struktur data/processed/nuscenes sesuai standar proyek.

Pastikan sudah install nuscenes-devkit, tqdm, dan pillow (PIL)!

Cara pakai:
$ python extract_nuscene.py
"""

import os
import shutil
import json
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
from PIL import Image

DATAROOT = '../../data/raw/nuscenes'
OUTPUT_ROOT = '../../data/processed/nuscenes'

CAMERAS = [
    'CAM_FRONT',
    'CAM_FRONT_LEFT',
    'CAM_FRONT_RIGHT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]
LIDARS = ['LIDAR_TOP']
RADARS = [
    'RADAR_FRONT',
    'RADAR_FRONT_LEFT',
    'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT',
    'RADAR_BACK_RIGHT'
]
IMG_SIZE = (640, 640)

def ensure_dirs():
    for cam in CAMERAS:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'camera', cam), exist_ok=True)
    for lidar in LIDARS:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'lidar', lidar), exist_ok=True)
    for radar in RADARS:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'radar', radar), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, 'annotations'), exist_ok=True)

def extract_sensor(nusc, sensor_list, modality, subfolder, ext='.pcd'):
    print(f"\nExtracting {modality} data...")
    count = 0
    for sd in tqdm(nusc.sample_data, desc=modality.capitalize()):
        if sd['sensor_modality'] == modality and sd['channel'] in sensor_list:
            src = os.path.join(DATAROOT, sd['filename'])
            dst = os.path.join(OUTPUT_ROOT, subfolder, sd['channel'], sd['token'] + ext)
            if not os.path.exists(src):
                print(f"WARNING: Source file not found: {src}")
                continue
            try:
                shutil.copyfile(src, dst)
                count += 1
            except Exception as e:
                print(f"ERROR: Failed to copy {src} to {dst}: {e}")
    print(f"Total {modality} files extracted: {count}")

def extract_images(nusc):
    print("\nExtracting & resizing camera images to 640x640...")
    count = 0
    for sd in tqdm(nusc.sample_data, desc="Camera"):
        if sd['sensor_modality'] == 'camera' and sd['channel'] in CAMERAS:
            src = os.path.join(DATAROOT, sd['filename'])
            dst = os.path.join(OUTPUT_ROOT, 'camera', sd['channel'], sd['token'] + '.jpg')
            if not os.path.exists(src):
                print(f"WARNING: Source file not found: {src}")
                continue
            try:
                img = Image.open(src)
                img = img.resize(IMG_SIZE, Image.BILINEAR)
                img.save(dst)
                count += 1
            except Exception as e:
                print(f"ERROR: Failed to process {src} to {dst}: {e}")
    print(f"Total camera images extracted & resized: {count}")

def extract_annotations(nusc):
    print("\nExtracting annotations...")
    anns = []
    for ann in tqdm(nusc.sample_annotation, desc="Annotations"):
        anns.append(ann)
    output_ann = os.path.join(OUTPUT_ROOT, 'annotations', 'sample_annotations.json')
    with open(output_ann, 'w') as f:
        json.dump(anns, f)
    print(f"Total annotations: {len(anns)}")

def main():
    ensure_dirs()
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=True)
    extract_images(nusc)  # kamera akan di-resize!
    extract_sensor(nusc, LIDARS, modality='lidar', subfolder='lidar', ext='.pcd')
    extract_sensor(nusc, RADARS, modality='radar', subfolder='radar', ext='.pcd')
    extract_annotations(nusc)
    print("\nExtract selesai! Data tersimpan di:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()