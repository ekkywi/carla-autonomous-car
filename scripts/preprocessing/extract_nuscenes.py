"""
Script untuk extract data mentah dari nuScenes ke folder lokal.
Contoh ini mengambil gambar kamera, point cloud lidar, radar, dan annotation (JSON)
dan menyimpannya ke struktur data/processed/nuscenes sesuai standar proyek.

Pastikan sudah install nuscenes-devkit dan tqdm!
"""

import os
import shutil
import json
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

# Path ke dataset nuScenes asli
DATAROOT = '../../data/raw/nuscenes'
# Path output data hasil extract
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

def ensure_dirs():
    # Buat folder output hanya sesuai sensor & kategori.
    for cam in CAMERAS:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'camera', cam), exist_ok=True)
    for lidar in LIDARS:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'lidar', lidar), exist_ok=True)
    for radar in RADARS:
        os.makedirs(os.path.join(OUTPUT_ROOT, 'radar', radar), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, 'annotations'), exist_ok=True)

def extract_images(nusc):
    print("\nExtracting camera images...")
    for sd in tqdm(nusc.sample_data, desc="Camera"):
        if sd['sensor_modality'] == 'camera' and sd['channel'] in CAMERAS:
            src = os.path.join(DATAROOT, sd['filename'])
            dst = os.path.join(OUTPUT_ROOT, 'camera', sd['channel'], sd['token'] + '.jpg')
            shutil.copyfile(src, dst)

def extract_lidar(nusc):
    print("\nExtracting lidar pointclouds...")
    for sd in tqdm(nusc.sample_data, desc="Lidar"):
        if sd['sensor_modality'] == 'lidar' and sd['channel'] in LIDARS:
            src = os.path.join(DATAROOT, sd['filename'])
            dst = os.path.join(OUTPUT_ROOT, 'lidar', sd['channel'], sd['token'] + '.pcd')
            shutil.copyfile(src, dst)

def extract_radar(nusc):
    print("\nExtracting radar data...")
    for sd in tqdm(nusc.sample_data, desc="Radar"):
        if sd['sensor_modality'] == 'radar' and sd['channel'] in RADARS:
            src = os.path.join(DATAROOT, sd['filename'])
            dst = os.path.join(OUTPUT_ROOT, 'radar', sd['channel'], sd['token'] + '.pcd')
            shutil.copyfile(src, dst)

def extract_annotations(nusc):
    print("\nExtracting annotations...")
    anns = []
    for ann in tqdm(nusc.sample_annotation, desc="Annotations"):
        anns.append(ann)
    with open(os.path.join(OUTPUT_ROOT, 'annotations', 'sample_annotations.json'), 'w') as f:
        json.dump(anns, f)
    print(f"Total annotations: {len(anns)}")

def main():
    ensure_dirs()
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=True)
    extract_images(nusc)
    extract_lidar(nusc)
    extract_radar(nusc)
    extract_annotations(nusc)
    print("\nExtract selesai! Data tersimpan di:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()