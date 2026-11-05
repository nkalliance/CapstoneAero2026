# Human Detection Drone – YOLO26

Deteksi manusia (1 kelas: `Person`) dari citra drone dengan YOLO26 (Ultralytics).
Dataset: 600 train / 25 valid / 25 test (sudah ada di folder `dataset/`).

## 1. Instalasi
```bash
pip install -r requirements.txt
# GPU NVIDIA: pasang PyTorch sesuai CUDA dulu dari https://pytorch.org
```

## 2. Training
```bash
python train.py                                   # yolo26n, 100 epoch, imgsz 640
python train.py --model yolo26s.pt --imgsz 960 --batch 8 --epochs 150
python train.py --device cpu --epochs 5           # uji cepat tanpa GPU
python train.py --resume                          # lanjutkan training terputus (otomatis cari last.pt terbaru)
python train.py --resume "C:/path/ke/runs/train/exp/weights/last.pt"   # atau tunjuk file langsung
```
Hasil: `runs/train/<name>/weights/best.pt` (+ grafik, confusion matrix, evaluasi val & test).
Karena manusia dari drone berukuran kecil, `--imgsz 960` atau `1280` biasanya menaikkan akurasi
(batch diturunkan jika VRAM kurang, atau `--batch -1` untuk otomatis).
Model bisa dipilih: `yolo26n/s/m/l/x.pt` (bobot pretrained terunduh otomatis).

## 3. Tes pada video
Sumber video diatur lewat `--source`:
```bash
python predict_video.py --weights runs/train/exp/weights/best.pt --source video.mp4
python predict_video.py --weights best.pt --source 0                  # webcam
python predict_video.py --weights best.pt --source rtsp://ip/stream   # RTSP / stream drone
python predict_video.py --weights best.pt --source folder_video/      # semua video dalam folder
python predict_video.py --weights best.pt --source v.mp4 --conf 0.3 --imgsz 960 --show
```
Opsi lain: `--iou`, `--device`, `--vid-stride N` (lewati frame biar cepat), `--no-save`,
`--output folder`, `--hide-labels`, `--hide-conf`, `--half`. Tekan **q** untuk menutup preview.
Hasil video + jumlah orang per frame disimpan di `runs/predict/`.
Gunakan `--imgsz` yang sama dengan saat training.

## Dataset
Dataset sudah termasuk di folder `dataset/` (train 600, valid 25, test 25).
Karena ukuran, paket dibagi 2 zip (batas upload 30 MB): ekstrak **kedua zip ke lokasi yang sama**
dan gabungkan foldernya (pilih "Merge"/"Replace" jika ditanya), supaya `dataset/train/images`
berisi 600 gambar. `dataset/data.yaml` sudah diperbaiki dari versi bawaan Roboflow.

Jika training lanjut error `images not found`, cek isi `dataset/train/images` dan `dataset/valid/images`.
