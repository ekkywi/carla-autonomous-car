import os
import shutil
import random
from tqdm import tqdm

def get_user_ratio():
    while True:
        try:
            ratio_str = input("Masukkan rasio split (train:val:test), misal 8:1:1 untuk 80% train, 10% val, 10% test (total 10): ")
            ratios = [int(x) for x in ratio_str.strip().split(":")]
            if len(ratios) != 3:
                raise ValueError("Rasio harus dalam format 3 angka, mis: 8:1:1")
            total = sum(ratios)
            if total != 10:
                raise ValueError("Jumlah rasio harus tepat 10, mis: 8:1:1")
            if any(r < 0 for r in ratios):
                raise ValueError("Rasio tidak boleh negatif.")
            return ratios, [r/total for r in ratios]
        except Exception as e:
            print(f"Input tidak valid: {e}\nContoh input yang benar: 8:1:1 (total 10)")

def main():
    IMAGE_SRC = '../../../data/processed/nuscenes/camera/images'
    LABEL_SRC = '../../../data/processed/nuscenes/camera/labels'
    OUT_ROOT = '../../../data/datasets/camera'
    SPLITS = ['train', 'val', 'test']

    print("=== Script Split Data YOLO (images & labels) ===\n")
    raw_ratios, ratios = get_user_ratio()
    print(f"\nRasio split: train={ratios[0]*100:.1f}%, val={ratios[1]*100:.1f}%, test={ratios[2]*100:.1f}%\n")

    all_images = sorted([f for f in os.listdir(IMAGE_SRC) if f.lower().endswith('.jpg')])
    random.shuffle(all_images)
    n = len(all_images)

    split_counts = [int(r * n) for r in ratios]
    # Koreksi agar total = n (misal saat pembulatan)
    split_counts[-1] = n - sum(split_counts[:-1])

    split_files = {}
    idx = 0
    for i, split in enumerate(SPLITS):
        count = split_counts[i]
        if count > 0:
            split_files[split] = all_images[idx:idx + count]
        idx += count

    # Hanya proses split dengan count > 0
    for split in SPLITS:
        if raw_ratios[SPLITS.index(split)] == 0:
            continue
        os.makedirs(os.path.join(OUT_ROOT, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, 'labels', split), exist_ok=True)

    print("Mulai menyalin file gambar & label ke folder split...\n")
    for split in SPLITS:
        if raw_ratios[SPLITS.index(split)] == 0:
            continue
        images_list = split_files.get(split, [])
        print(f"Split '{split}' ({len(images_list)} file):")
        for img_file in tqdm(images_list, desc=f"  Copying {split}", unit='file'):
            label_file = img_file.replace('.jpg', '.txt')
            src_img = os.path.join(IMAGE_SRC, img_file)
            dst_img = os.path.join(OUT_ROOT, 'images', split, img_file)
            shutil.copy2(src_img, dst_img)
            src_lbl = os.path.join(LABEL_SRC, label_file)
            dst_lbl = os.path.join(OUT_ROOT, 'labels', split, label_file)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                open(dst_lbl, 'a').close()

    print("\n=== Selesai split data ke folder images/ dan labels/ per split ===")
    print("  Lokasi contoh hasil:")
    for split in SPLITS:
        if raw_ratios[SPLITS.index(split)] == 0:
            continue
        print(f"    images/{split}/, labels/{split}/")
    print()

if __name__ == "__main__":
    main()