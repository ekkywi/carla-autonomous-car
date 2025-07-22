"""
Script untuk konversi file point cloud radar nuScenes (.pcd) menjadi gambar heatmap dan scatter berukuran 640x640 pixel,
agar hasil konsisten dengan pipeline YOLO.

- Input: folder berisi file radar (.pcd) hasil ekstrak dari nuScenes
- Output: gambar heatmap (.png) dan scatter (.png) per file radar di folder output, ukuran 640x640 pixel
- Struktur output: data/processed/nuscenes/radar/radar_heatmap_img/<RADAR_CHANNEL>/

Pastikan sudah install numpy, matplotlib, pillow (PIL)!

Cara pakai:
$ python radar_to_image.py
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

RADARS = [
    'RADAR_FRONT',
    'RADAR_FRONT_LEFT',
    'RADAR_FRONT_RIGHT',
    'RADAR_BACK_LEFT',
    'RADAR_BACK_RIGHT'
]

RADAR_ROOT = '../../data/processed/nuscenes/radar'
OUTPUT_ROOT = '../../data/processed/nuscenes/radar/radar_heatmap_img'

# Radar nuScenes format
FIELDS = [
    "x", "y", "z", "dyn_prop", "id", "rcs", "vx", "vy", "vx_comp", "vy_comp",
    "is_quality_valid", "ambig_state", "x_rms", "y_rms", "invalid_state",
    "pdh0", "vx_rms", "vy_rms"
]
SIZES = [4, 4, 4, 1, 2, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1]
TYPES = ["f", "f", "f", "b", "h", "f", "f", "f", "f", "f", "b", "b", "b", "b", "b", "b", "b", "b"]
POINT_SIZE = sum(SIZES)

IMG_SIZE = (640, 640)  # Output image size

def build_struct_fmt(sizes, types):
    fmt = "<"
    for size, type_ in zip(sizes, types):
        if type_ == "f":
            fmt += "f"
        elif type_ == "b":
            fmt += "b"
        elif type_ == "h":
            fmt += "h"
    return fmt

def read_radar_pcd(filename):
    with open(filename, 'rb') as f:
        # Skip header (read until DATA binary)
        while True:
            line = f.readline()
            if b'DATA binary' in line:
                break
        raw = f.read()
        n_points = len(raw) // POINT_SIZE
        fmt = build_struct_fmt(SIZES, TYPES)
        points = []
        for i in range(n_points):
            chunk = raw[i*POINT_SIZE:(i+1)*POINT_SIZE]
            if len(chunk) != POINT_SIZE:
                continue
            point = struct.unpack(fmt, chunk)
            points.append(point)
        return np.array(points)

def resize_and_save(img_path, out_path, size=IMG_SIZE):
    img = Image.open(img_path)
    img = img.resize(size, Image.BILINEAR)
    img.save(out_path)

def viz_heatmap(x, y, out_path):
    plt.figure(figsize=(8,8))
    plt.hexbin(x, y, gridsize=40, cmap='hot', mincnt=1)
    plt.colorbar()
    plt.title('Radar Heatmap')
    plt.xlabel('X (meter)')
    plt.ylabel('Y (meter)')
    plt.tight_layout()
    tmp_path = out_path.replace('.png', '_tmp.png')
    plt.savefig(tmp_path)
    plt.close()
    resize_and_save(tmp_path, out_path, IMG_SIZE)
    os.remove(tmp_path)

def viz_scatter(x, y, out_path):
    plt.figure(figsize=(8,8))
    plt.scatter(x, y, s=50, c='red', alpha=0.7)
    plt.title('Radar Scatter')
    plt.xlabel('X (meter)')
    plt.ylabel('Y (meter)')
    plt.tight_layout()
    tmp_path = out_path.replace('.png', '_tmp.png')
    plt.savefig(tmp_path)
    plt.close()
    resize_and_save(tmp_path, out_path, IMG_SIZE)
    os.remove(tmp_path)

def process_radar_channel(channel_dir, output_dir):
    files = [f for f in os.listdir(channel_dir) if f.endswith('.pcd')]
    if not files:
        print(f"[WARNING] Tidak ditemukan file .pcd di {channel_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    for fname in files:
        fpath = os.path.join(channel_dir, fname)
        points = read_radar_pcd(fpath)
        if points.shape[0] == 0:
            print(f"[WARNING] File {fname} kosong/tidak valid. Skip.")
            continue
        x = points[:,0]
        y = points[:,1]
        print(f"[{os.path.basename(channel_dir)}] X min/max: {x.min():.2f} / {x.max():.2f}")
        print(f"[{os.path.basename(channel_dir)}] Y min/max: {y.min():.2f} / {y.max():.2f}")
        print(f"[{os.path.basename(channel_dir)}] Jumlah point: {len(x)}")
        out_heatmap = os.path.join(output_dir, fname.replace('.pcd', '_heatmap.png'))
        out_scatter = os.path.join(output_dir, fname.replace('.pcd', '_scatter.png'))
        viz_heatmap(x, y, out_heatmap)
        viz_scatter(x, y, out_scatter)
        print(f"[INFO] Saved: {out_heatmap}, {out_scatter}")

def main():
    for channel in RADARS:
        channel_dir = os.path.join(RADAR_ROOT, channel)
        output_dir = os.path.join(OUTPUT_ROOT, channel)
        print(f"\n[INFO] Processing channel: {channel}")
        process_radar_channel(channel_dir, output_dir)
    print("\n[INFO] Selesai proses semua channel radar.")

if __name__ == "__main__":
    main()