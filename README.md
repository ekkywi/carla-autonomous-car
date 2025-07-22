# 🚗 CARLA Autonomous Car Project

**Simulasi dan Pengembangan Sistem Kendaraan Otonom Menggunakan CARLA Simulator**

---

## 📌 Deskripsi Proyek

Proyek ini adalah framework pengembangan **kendaraan otonom (autonomous car)** berbasis simulasi menggunakan [CARLA Simulator](https://carla.org/). Sistem mengintegrasikan data dari berbagai sensor seperti **kamera, LiDAR, dan radar**, serta mendukung penggunaan dataset eksternal seperti **[nuScenes](https://www.nuscenes.org/)** untuk pelatihan dan pengujian model.

> Pengembangan menggunakan **Python 3.8.18**

---

## 📦 Fitur Utama

- ✅ Integrasi sensor kamera, LiDAR, dan radar dalam simulasi CARLA
- ✅ Workflow untuk preprocessing dan visualisasi data sensor
- ✅ Dukungan dataset eksternal (seperti nuScenes)
- ✅ Framework modular untuk training dan inference model autonomous
- ✅ Kompatibel dengan pipeline Machine Learning dan Deep Learning

---

## 🧰 Teknologi & Tools

- [CARLA Simulator](https://carla.org/) (0.9.x)
- Python 3.8.18
- PyTorch / TensorFlow (opsional, untuk model ML/DL)
- OpenCV, NumPy, Matplotlib
- nuScenes Dataset Tools

---

## 📁 Struktur Direktori

```
carla-autonomous-car/
├── README.md                         # Dokumentasi utama proyek
├── requirements.txt                  # Daftar dependensi pip
├── environment.yml                   # File environment Conda (opsional)
│
├── config/                           # 🔧 File konfigurasi simulasi dan training
│   ├── config_camera.yaml
│   ├── config_lidar.yaml
│   ├── config_radar.yaml
│   └── class_mapping_nuscenes.yaml   # Mapping class ke ID numerik dari data NuScenes
│
├── data/                             # 📦 Semua data (mentah, olahan, training)
│   ├── raw/                          # 📂 Data asli tanpa modifikasi
│   │   ├── nuscenes/                 # Dataset asli nuScenes
│   │   └── carla/                    # Hasil sim CARLA mentah (gambar, pcd, radar)
│   │
│   ├── processed/                    # 📂 Data hasil preprocessing
│   │   ├── nuscenes/
│   │   │   ├── camera/
│   │   │   ├── lidar/
│   │   │   ├── radar/
│   │   │   └── annotations/         # Bounding box, metadata
│   │   └── carla/
│   │       ├── camera/
│   │       ├── lidar/
│   │       ├── radar/
│   │       └── annotations/
│   │
│   ├── datasets/                     # 📂 Dataset siap training, dipisah per sensor
│   │   ├── camera/                   # Dataset dari kamera (RGB)
│   │   │   ├── images/
│   │   │   │   ├── train/
│   │   │   │   └── val/
│   │   │   └── labels/
│   │   │       ├── train/
│   │   │       └── val/
│   │   │
│   │   ├── lidar/                    # Dataset dari LiDAR (point cloud)
│   │   │   ├── pointclouds/
│   │   │   │   ├── train/
│   │   │   │   └── val/
│   │   │   └── labels/
│   │   │       ├── train/
│   │   │       └── val/
│   │   │
│   │   ├── radar/                    # Dataset dari radar
│   │   │   ├── raw/
│   │   │   │   ├── train/
│   │   │   │   └── val/
│   │   │   └── labels/
│   │   │       ├── train/
│   │   │       └── val/
│   │   │
│   │   └── fusion/                   # Dataset untuk model fusi multi-sensor
│   │       ├── inputs/              # Gabungan kamera + lidar + radar per frame
│   │       └── labels/
│   │           ├── train/
│   │           └── val/
│   │
│   ├── visualizations/              # 📊 Hasil visualisasi (bounding box, heatmap, dsb)
│   │   ├── camera/
│   │   ├── lidar/
│   │   ├── radar/
│   │   └── fusion/
│   │
│   └── meta/                        # 📑 Metadata global
│       ├── train_val_split.json
│       └── sensor_calibration.json
│
├── models/                          # 🧠 Model hasil training
│   ├── yolov8_camera.pt
│   ├── yolov8_lidar.pt
│   ├── yolov8_radar.pt
│   └── fusion_model.pt
│
├── scripts/                         # ⚙️ Script Python untuk semua tahapan
│   ├── preprocessing/
│   │   ├── extract_nuscenes.py
│   │   ├── preprocess_carla.py
│   │   └── convert_to_yolo.py
│   │
│   ├── training/
│   │   └── train_yolo.py
│   │
│   ├── fusion/
│   │   └── build_fusion_dataset.py
│   │
│   └── visualization/
│       └── visualize_bboxes.py
│
├── carla_api/                       # 🔌 Komunikasi dengan CARLA simulator
│   ├── client.py
│   ├── sensor.py
│   ├── vehicle.py
│   └── utils.py

```

---

## 🚀 Instalasi & Penggunaan

### 1. Clone Repository

```bash
git clone https://github.com/username/carla-autonomous-car.git
cd carla-autonomous-car
```

### 2. Buat Virtual Environment

```bash
python3.8 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📚 Dataset Support

Framework ini mendukung integrasi dengan dataset nuScenes.  
Silakan unduh dataset dari: https://www.nuscenes.org/download

Letakkan dataset di dalam folder `data/nuscenes/` dan ikuti petunjuk preprocessing di notebooks/.

---

## ▶️ Menjalankan Simulasi

Contoh menjalankan skrip simulasi:

```bash
python scripts/run_simulation.py --sensor-config config/config.yaml
```

---

## 📈 Roadmap Pengembangan

- Integrasi sensor CARLA
- Visualisasi sensor multi-view
- Integrasi real-time inference
- Pelatihan model berbasis nuScenes
- Integrasi ROS2 (opsional)

---

## 🤝 Kontribusi

Kontribusi terbuka untuk siapa saja.  
Silakan fork repo ini dan ajukan pull request.

---

## 🛡️ Lisensi

Proyek ini dirilis di bawah lisensi MIT.  
Lihat file LICENSE untuk informasi lebih lanjut.

---

## 📞 Kontak

Dikembangkan oleh:  
**Yon Ekky Wijayanto**  
📧 yonekkywijayanto@outlook.com
🌐 [LinkedIn](https://www.linkedin.com/in/yon-ekky-wijayanto-7008ab295/)
