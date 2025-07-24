import os

IMAGE_DIR = '../../../data/processed/nuscenes/camera/images'
LABEL_DIR = '../../../data/processed/nuscenes/camera/labels'

num_images = 0
num_labels = 0
num_pairs = 0
num_missing_label = 0
num_empty_label = 0
num_invalid_format = 0

invalid_files = []

def check_label_format(line):
    """
    Cek format 1 baris label YOLO:
    - Harus terdiri dari 5 kolom (class_id, x_center, y_center, width, height)
    - Semua numerik (class_id boleh int/float)
    - x_center, y_center, width, height di [0,1]
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return False
    try:
        cls = int(float(parts[0]))
        vals = [float(x) for x in parts[1:]]
        if not all(0.0 <= v <= 1.0 for v in vals):
            return False
    except Exception:
        return False
    return True

all_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.jpg')]

for img_file in all_images:
    num_images += 1
    label_file = img_file.replace('.jpg', '.txt')
    label_path = os.path.join(LABEL_DIR, label_file)
    if not os.path.exists(label_path):
        num_missing_label += 1
        invalid_files.append((img_file, "File label tidak ditemukan"))
        continue
    num_labels += 1
    with open(label_path, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0:
        num_empty_label += 1
        invalid_files.append((img_file, "File label kosong"))
        continue
    valid = True
    for line in lines:
        if not check_label_format(line):
            num_invalid_format += 1
            invalid_files.append((img_file, f"Format label tidak valid: {line.strip()}"))
            valid = False
            break
    if valid:
        num_pairs += 1

print("\n=== Verifikasi Data Kamera ===")
print(f"Total gambar ditemukan      : {len(all_images)}")
print(f"Total label ditemukan       : {num_labels}")
print(f"Pasangan gambar-label valid : {num_pairs}")
print(f"Label hilang                : {num_missing_label}")
print(f"File label kosong           : {num_empty_label}")
print(f"Invalid label format        : {num_invalid_format}")
print("\n=== Detail Verifikasi ===")

total_invalid = num_missing_label + num_empty_label + num_invalid_format
total_checked = num_pairs + total_invalid
print(f"Total pasangan dicek        : {total_checked}")
print(f"Total pasangan tidak valid  : {total_invalid}\n")

if total_invalid > 0:
    print("Ada masalah pada beberapa pasangan image-label:")
    print(f" - {num_missing_label} label hilang")
    print(f" - {num_empty_label} label kosong")
    print(f" - {num_invalid_format} label dengan format tidak valid")
    if invalid_files:
        print("\nDaftar file bermasalah :")
        for f, err in invalid_files[:]:
            print(f" - {f}: {err}")
    print("\nMohon periksa dan perbaiki sebelum melanjutkan ke training YOLO.\n")
else:
    print("Semua pasangan image-label valid! Siap untuk training YOLO.\n")