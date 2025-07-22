"""
lidar_to_image.py

Script untuk mengkonversi point cloud LIDAR nuScenes (.pcd atau .bin) menjadi citra BEV (Bird's Eye View).
- Input: folder berisi file point cloud LIDAR (.pcd atau .bin) hasil ekstrak dari nuScenes
- Output: gambar BEV (.png) per file LIDAR di folder output

Format file lidar nuScenes:
- .bin: float32, tiap baris [x, y, z, intensity]
- .pcd: biasanya format bin juga, tanpa header PCD (bukan standar). Diproses sama seperti .bin.

Dependencies:
- numpy
- pillow (`PIL`)
- (tidak perlu open3d!)

Cara pakai:
$ python lidar_to_image.py
"""

import os
import numpy as np
from PIL import Image

INPUT_LIDAR_DIR = '../../data/processed/nuscenes/lidar/LIDAR_TOP'
OUTPUT_IMG_DIR = '../../data/processed/nuscenes/lidar/lidar_bev_img'

BEV_RES = 0.15625  # (X_MAX-X_MIN)/RES = 100/0.15625 = 640
X_MIN, X_MAX = -50, 50
Y_MIN, Y_MAX = -50, 50
IMG_SIZE = (640, 640)

def pointcloud_to_bev(points, x_min, x_max, y_min, y_max, res):
    mask = (points[:, 0] > x_min) & (points[:, 0] < x_max) & (points[:, 1] > y_min) & (points[:, 1] < y_max)
    pc = points[mask]
    x_bins = int((x_max - x_min) / res)
    y_bins = int((y_max - y_min) / res)
    bev_img = np.zeros((y_bins, x_bins), dtype=np.uint8)
    if pc.shape[0] > 0:
        x_indices = ((pc[:, 0] - x_min) / res).astype(int)
        y_indices = ((pc[:, 1] - y_min) / res).astype(int)
        valid = (x_indices >= 0) & (x_indices < x_bins) & (y_indices >= 0) & (y_indices < y_bins)
        bev_img[y_indices[valid], x_indices[valid]] = 255
    else:
        print("[WARNING] BEV empty for current file.")
    bev_img = np.flipud(bev_img)
    return bev_img

def process_lidar_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('.bin') or f.endswith('.pcd')]
    if not files:
        print(f"[WARNING] Tidak ditemukan file .bin/.pcd di {input_dir}")
    for fname in files:
        fpath = os.path.join(input_dir, fname)
        try:
            points = np.fromfile(fpath, dtype=np.float32)
            if points.shape[0] % 4 != 0:
                print(f"[WARNING] File {fname} size not divisible by 4. Skip.")
                continue
            points = points.reshape(-1, 4)[:, :3]
            print(f"[INFO] {fname}: {points.shape[0]} points")
        except Exception as e:
            print(f"[ERROR] Failed to read {fname}: {e}")
            continue
        bev_img = pointcloud_to_bev(points, X_MIN, X_MAX, Y_MIN, Y_MAX, BEV_RES)
        # Output BEV sudah 640x640, tidak perlu resize lagi
        out_imgname = "bev_" + fname.replace('.bin', '.png').replace('.pcd', '.png')
        out_imgpath = os.path.join(output_dir, out_imgname)
        Image.fromarray(bev_img).save(out_imgpath)
        print(f"[INFO] Saved BEV image: {out_imgpath}")

if __name__ == "__main__":
    process_lidar_folder(INPUT_LIDAR_DIR, OUTPUT_IMG_DIR)