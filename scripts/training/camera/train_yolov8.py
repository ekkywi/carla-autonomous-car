import os
import time
import pandas as pd
import numpy as np
from ultralytics import YOLO

def clear():
    os.system('cls' if os.name == 'nt' else 'clear')

def input_with_default(prompt, default, desc):
    print(f"\n{prompt}")
    print(f"  Fungsi: {desc}")
    print(f"  Default: {default} (tekan ENTER untuk memakai default)")
    user = input("  > ").strip()
    return user if user else default

def select_from_list(prompt, options, default, desc):
    print(f"\n{prompt}")
    print(f"  Fungsi: {desc}")
    for i, (k, v) in enumerate(options.items()):
        print(f"    {i+1}. {v} ({k})")
    print(f"  Default: {default} (tekan ENTER untuk memakai default)")
    while True:
        user = input("  > ").strip()
        if not user:
            return default
        try:
            idx = int(user) - 1
            if 0 <= idx < len(options):
                return list(options.keys())[idx]
        except ValueError:
            if user in options:
                return user
        print("  Input tidak valid.")

def print_section(title):
    print("="*len(title))
    print(title)
    print("="*len(title))

def safe_float_fmt(val, fmt=".2f"):
    try:
        if val is None or val == "?":
            return "?"
        return format(float(val), fmt)
    except Exception:
        return str(val)

def print_training_summary(results, val_results, model_dir, start_time, end_time):
    # Info model & kelas
    names = getattr(val_results, "names", None)
    nc = len(names) if names else "?"
    model_param = "?"
    try:
        model_obj = getattr(val_results, "model", None)
        if model_obj:
            model_param = sum(p.numel() for p in model_obj.parameters()) / 1e6
    except Exception:
        pass

    # Epoch terbaik & waktu training
    csv_path = os.path.join(model_dir, "results.csv")
    best_epoch = best_map = None
    epoch_time = "-"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            best_idx = df['metrics/mAP_0.5'].idxmax()
            best_epoch = int(df.iloc[best_idx]['epoch'])
            best_map = df.iloc[best_idx]['metrics/mAP_0.5']
            total_time = df['epoch_time'].sum() if 'epoch_time' in df else None
            if total_time:
                epoch_time = f"{total_time/3600:.2f} jam ({total_time/60:.0f} menit)"
            else:
                elapsed = end_time - start_time
                epoch_time = f"{elapsed/3600:.2f} jam ({elapsed/60:.0f} menit)"
        except Exception:
            pass

    # Metrik utama dari val_results
    box = getattr(val_results, "box", None)
    map50 = safe_float_fmt(getattr(box, "map50", "?")) if box else "?"
    map5095 = safe_float_fmt(getattr(box, "map", "?")) if box else "?"
    try:
        precision = safe_float_fmt(np.mean(box.precision)) if box and hasattr(box, "precision") else "?"
        recall = safe_float_fmt(np.mean(box.recall)) if box and hasattr(box, "recall") else "?"
    except Exception:
        precision = safe_float_fmt(box.precision.mean()) if box and hasattr(box, "precision") and hasattr(box.precision, "mean") else "?"
        recall = safe_float_fmt(box.recall.mean()) if box and hasattr(box, "recall") and hasattr(box.recall, "mean") else "?"

    # Inference speed
    speed = getattr(val_results, "speed", {}).get("inference", "?")
    try:
        speed_str = f"{float(speed):.1f} ms/gambar"
    except Exception:
        speed_str = f"{speed} ms/gambar"

    # File model terbaik
    best_pt = os.path.join(model_dir, "weights", "best.pt")

    print("\n=== RINGKASAN HASIL TRAINING YOLOv8 ===")
    print(f"- Jumlah kelas         : {nc}         # Banyaknya kelas deteksi")
    print(f"- Model parameter      : {safe_float_fmt(model_param, '.1f')}M # Ukuran model (jutaan parameter)")
    print(f"- Epoch terbaik        : {best_epoch if best_epoch is not None else '-'} (mAP50: {safe_float_fmt(best_map) if best_map is not None else '-'})")
    print(f"- Total waktu training : {epoch_time}")
    print(f"- Inference speed      : {speed_str}")
    print(f"- mAP50 (IoU 0.5)      : {map50}")
    print(f"- mAP50-95 (COCO)      : {map5095}")
    print(f"- Precision rata-rata  : {precision}")
    print(f"- Recall rata-rata     : {recall}")
    print(f"- File model terbaik   : {best_pt}")
    print("\nKeterangan:")
    print("• mAP50 >0.5: Sudah cukup baik untuk deteksi objek.")
    print("• Precision: Persentase prediksi benar.")
    print("• Recall: Persentase objek yang berhasil terdeteksi.")
    print("• Jika mAP50 <0.1, cek dataset/label atau tambah epoch training.")
    print("="*40 + "\n")

def main():
    clear()
    print_section("YOLOv8 Training Interaktif")
    print(
        "Script ini akan membantu Anda menyiapkan training YOLOv8 secara interaktif.\n"
        "Setiap pertanyaan disertai penjelasan fungsinya.\n"
        "Tekan ENTER untuk memakai nilai default.\n"
    )

    # Pilihan model
    model_map = {
        "n": "YOLOv8 Nano   (kecil, sangat ringan)",
        "s": "YOLOv8 Small  (kecil, cukup ringan)",
        "m": "YOLOv8 Medium (sedang, akurasi bagus)",
        "l": "YOLOv8 Large  (besar, akurasi tinggi)",
        "x": "YOLOv8 XLarge (paling besar, sangat akurat)",
        "custom": "Model Custom (.pt sendiri)"
    }
    model_choice = select_from_list(
        "Pilih model YOLOv8:",
        model_map,
        "s",
        "Menentukan arsitektur model YOLOv8 yang akan digunakan. 'n' paling ringan, 'x' paling akurat (tapi berat)."
    )
    if model_choice == "custom":
        model = input_with_default(
            "Masukkan path file model custom (.pt):",
            "./models/yolov8s.pt",
            "Path ke file model .pt yang sudah Anda siapkan sendiri."
        )
    else:
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        os.makedirs(models_dir, exist_ok=True)
        model = os.path.join(models_dir, f"yolov8{model_choice}.pt")
        if not os.path.exists(model):
            print(f">>> Model {model} belum ada, akan otomatis diunduh oleh YOLO.")

    # Path data.yaml
    dataset_yaml = input_with_default(
        "Masukkan path ke data.yaml:",
        "../../../data/datasets/camera/data.yaml",
        "File konfigurasi dataset: lokasi data gambar/label dan daftar kelas."
    )

    # Epoch
    while True:
        try:
            epochs = int(input_with_default(
                "Epoch (berapa kali seluruh data akan dilatih):",
                "50",
                "Jumlah iterasi seluruh dataset dilatih. 50–100 umumnya cukup untuk dataset kecil/menengah."
            ))
            if epochs > 0:
                break
            print("  Input harus angka positif.")
        except ValueError:
            print("  Input harus angka.")

    # Image size
    while True:
        try:
            imgsz = int(input_with_default(
                "Ukuran gambar (imgsz, kelipatan 32):",
                "640",
                "Resolusi input gambar ke model. Semakin besar, semakin akurat, tapi butuh memori lebih banyak."
            ))
            if imgsz >= 128 and imgsz % 32 == 0:
                break
            print("  Input harus >=128 dan kelipatan 32.")
        except ValueError:
            print("  Input harus angka.")

    # Batch size
    while True:
        try:
            batch = int(input_with_default(
                "Batch size (jumlah gambar per langkah training):",
                "16",
                "Jumlah gambar diproses sekaligus per iterasi. 16 cocok untuk GPU 8GB. Sesuaikan dengan memori."
            ))
            if batch > 0:
                break
            print("  Input harus angka positif.")
        except ValueError:
            print("  Input harus angka.")

    # Learning rate
    lr = input_with_default(
        "Learning rate (kosongkan untuk default):",
        "",
        "Kecepatan perubahan bobot saat training. Kosongkan untuk pakai nilai otomatis."
    )

    # Device
    device_def = "0" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu"
    device = input_with_default(
        "Device (0=GPU, cpu=CPU):",
        device_def,
        "Menentukan perangkat pemrosesan. 0 (default) untuk GPU pertama, 'cpu' untuk CPU."
    )

    # Workers
    while True:
        try:
            workers = int(input_with_default(
                "Workers (proses paralel data loading):",
                "4",
                "Jumlah proses paralel untuk membaca data. 2-8 umum, 0 jika error di Windows."
            ))
            if workers >= 0:
                break
            print("  Input harus >=0.")
        except ValueError:
            print("  Input harus angka.")

    # Ringkasan
    clear()
    print_section("Ringkasan Konfigurasi Training")
    print(f"Model        : {model} {'(custom)' if model_choice == 'custom' else ''}")
    print(f"Dataset yaml : {dataset_yaml}")
    print(f"Epochs       : {epochs}")
    print(f"Img size     : {imgsz}")
    print(f"Batch size   : {batch}")
    print(f"Learning rate: {'default' if not lr else lr}")
    print(f"Device       : {device}")
    print(f"Workers      : {workers}")
    print("\nLanjutkan training? (Y/n)")
    confirm = input("  > ").strip().lower()
    if confirm == "n":
        print("Dibatalkan.")
        return

    # Tentukan project_dir (history tetap, tidak pernah dihapus)
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../runs/detect/camera"))

    print("\nMemulai training YOLOv8 ...\n")
    model_obj = YOLO(model)
    train_kwargs = dict(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project_dir,
        # name dikosongkan supaya Ultralytics auto-buat train, train2, dst
        name="",
        workers=workers,
        device=device,
        amp=False
    )
    if lr:
        try:
            train_kwargs["lr0"] = float(lr)
        except Exception:
            print("Learning rate tidak valid, gunakan default.")

    start_time = time.time()
    results = model_obj.train(**train_kwargs)
    end_time = time.time()
    print("\nTraining selesai!")

    # Cari folder training terbaru (train, train2, dst) untuk summary dan validasi
    try:
        # Cari semua folder yang diawali 'train'
        candidates = [d for d in os.listdir(project_dir) if d.startswith("train")]
        if candidates:
            runs_path = os.path.join(project_dir, sorted(candidates, key=lambda x: int(''.join(filter(str.isdigit, x))) if any(i.isdigit() for i in x) else 0)[-1])
        else:
            runs_path = os.path.join(project_dir, "train")

        # Jalankan validasi ulang pada model terbaik, output ke folder yang sama agar tidak buat train baru
        weights_path = os.path.join(runs_path, "weights", "best.pt")
        val_results = model_obj.val(model=weights_path, project=project_dir, name=os.path.basename(runs_path), exist_ok=True, amp=False)

        print_training_summary(results, val_results, runs_path, start_time, end_time)
    except Exception as e:
        print("Gagal menampilkan ringkasan hasil training:", e)

if __name__ == "__main__":
    main()