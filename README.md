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
├── README.md
├── requirements.txt
├── environment.yml                             # opsional, jika pakai conda
├── data/
│   ├── nuscenes/                               # dataset nuScenes asli
│   ├── carla_images/                           # hasil capture kamera CARLA
│   ├── yolo_dataset/                           # dataset siap training YOLOv8
│   └── yolo_labels/                            # labeling YOLO
├── models/
│   └── best.pt                                 # model YOLOv8 hasil training
├── scripts/
│   ├── generate_nuscnenes_image_mapping.py     # generate image mapping
│   ├── generate_yolo_labels_nuscenes_3.py      # generate YOLO label
│   ├── split_and_copy_yolo_nuscenes.py         # split data persiapan train YOLO
│   └── train.py                                # train YOLOv8
├── carla_api/
│   ├── client.py
│   ├── sensor.py
│   ├── vehicle.py
│   └── utils.py
└── config/
    └── config.yaml                             # konfigurasi simulasi, model, sensor
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
📧 ekky@example.com  
🌐 [LinkedIn](#)
