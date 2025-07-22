"""
Script untuk verifikasi SELURUH hasil extract data nuScenes.
- Mengecek semua file gambar kamera, point cloud lidar, radar, dan annotation.
- Melaporkan file corrupt/missing.
- Menampilkan info distribusi annotation.

Pastikan sudah install: pillow, tqdm, json
"""

import os
import json
from PIL import Image
from tqdm import tqdm

OUTPUT_ROOT = '../../data/processed/nuscenes'

def verify_images():
    print("\nVerifikasi semua gambar kamera:")
    cameras_root = os.path.join(OUTPUT_ROOT, 'camera')
    for cam in os.listdir(cameras_root):
        cam_dir = os.path.join(cameras_root, cam)
        if not os.path.isdir(cam_dir):
            continue  # skip file seperti .gitkeep
        files = os.listdir(cam_dir)
        print(f"  {cam}: {len(files)} file")
        error_count = 0
        for fname in tqdm(files, desc=f"Cek {cam}"):
            fpath = os.path.join(cam_dir, fname)
            try:
                img = Image.open(fpath)
                img.verify()
            except Exception as e:
                error_count += 1
                print(f"    ERROR: {fname} ({e})")
        if error_count == 0:
            print(f"    Semua file OK untuk {cam}")
        else:
            print(f"    Jumlah file bermasalah di {cam}: {error_count}")

def verify_lidar():
    print("\nVerifikasi semua point cloud lidar:")
    lidar_root = os.path.join(OUTPUT_ROOT, 'lidar')
    for lidar in os.listdir(lidar_root):
        lidar_dir = os.path.join(lidar_root, lidar)
        if not os.path.isdir(lidar_dir):
            continue  # skip file seperti .gitkeep
        files = os.listdir(lidar_dir)
        print(f"  {lidar}: {len(files)} file")
        error_count = 0
        for fname in tqdm(files, desc=f"Cek {lidar}"):
            fpath = os.path.join(lidar_dir, fname)
            if not os.path.exists(fpath):
                error_count += 1
                print(f"    ERROR: {fname} (file not found)")
            elif os.path.getsize(fpath) == 0:
                error_count += 1
                print(f"    ERROR: {fname} (file size 0)")
        if error_count == 0:
            print(f"    Semua file OK untuk {lidar}")
        else:
            print(f"    Jumlah file bermasalah di {lidar}: {error_count}")

def verify_radar():
    print("\nVerifikasi semua point cloud radar:")
    radar_root = os.path.join(OUTPUT_ROOT, 'radar')
    for radar in os.listdir(radar_root):
        radar_dir = os.path.join(radar_root, radar)
        if not os.path.isdir(radar_dir):
            continue  # skip file seperti .gitkeep
        files = os.listdir(radar_dir)
        print(f"  {radar}: {len(files)} file")
        error_count = 0
        for fname in tqdm(files, desc=f"Cek {radar}"):
            fpath = os.path.join(radar_dir, fname)
            if not os.path.exists(fpath):
                error_count += 1
                print(f"    ERROR: {fname} (file not found)")
            elif os.path.getsize(fpath) == 0:
                error_count += 1
                print(f"    ERROR: {fname} (file size 0)")
        if error_count == 0:
            print(f"    Semua file OK untuk {radar}")
        else:
            print(f"    Jumlah file bermasalah di {radar}: {error_count}")

def verify_annotations():
    print("\nVerifikasi seluruh annotation:")
    ann_file = os.path.join(OUTPUT_ROOT, 'annotations', 'sample_annotations.json')
    if os.path.exists(ann_file):
        with open(ann_file) as f:
            anns = json.load(f)
        print(f"  Total annotation: {len(anns)}")
        # Distribusi category
        categories = {}
        for ann in anns:  # seluruh annotation
            cat = ann.get('category_name', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1
        print("  Distribusi kategori:")
        for k, v in categories.items():
            print(f"    {k}: {v}")
        if len(anns) > 0:
            print("  Contoh annotation:", anns[0])
    else:
        print("  ERROR: File annotation tidak ditemukan!")

def main():
    verify_images()
    verify_lidar()
    verify_radar()
    verify_annotations()
    print("\nVerifikasi selesai.")

if __name__ == "__main__":
    main()