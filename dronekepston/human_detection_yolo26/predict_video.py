"""Uji model YOLO26 pada video / webcam / stream, sumber bisa diatur lewat --source.

Contoh:
    python predict_video.py --weights best.pt --source video.mp4
    python predict_video.py --weights best.pt --source 0                 # webcam
    python predict_video.py --weights best.pt --source rtsp://ip/stream  # kamera/drone stream
    python predict_video.py --weights best.pt --source folder_video/     # semua video di folder
    python predict_video.py --weights best.pt --source video.mp4 --conf 0.3 --imgsz 960 --show
    python predict_video.py --weights best.pt --source video.mp4 --no-save --show

Tekan 'q' pada jendela preview untuk berhenti.
"""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}


def parse_args():
    p = argparse.ArgumentParser(description="Tes model YOLO26 pada video")
    p.add_argument("--weights", default="auto",
                   help="path bobot (.pt). 'auto' = best.pt terbaru di runs/train")
    p.add_argument("--source", required=True,
                   help="file video, folder berisi video, indeks webcam (0,1,..), atau URL rtsp/http")
    p.add_argument("--output", default=str(ROOT / "runs" / "predict"), help="folder hasil")
    p.add_argument("--conf", type=float, default=0.25, help="ambang confidence")
    p.add_argument("--iou", type=float, default=0.7, help="ambang IoU NMS")
    p.add_argument("--imgsz", type=int, default=640, help="ukuran inferensi (samakan/naikkan dari training)")
    p.add_argument("--device", default=None, help="'0', 'cpu', 'mps' (kosong = otomatis)")
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--vid-stride", type=int, default=1, help="proses tiap N frame (percepat di CPU)")
    p.add_argument("--show", action="store_true", help="tampilkan jendela preview")
    p.add_argument("--no-save", action="store_true", help="jangan simpan video hasil")
    p.add_argument("--hide-labels", action="store_true")
    p.add_argument("--hide-conf", action="store_true")
    p.add_argument("--line-width", type=int, default=None)
    p.add_argument("--half", action="store_true", help="FP16 (GPU)")
    return p.parse_args()


def list_sources(src: str):
    if src.isdigit():
        return [int(src)]
    path = Path(src)
    if path.is_dir():
        vids = sorted(f for f in path.iterdir() if f.suffix.lower() in VIDEO_EXT)
        if not vids:
            raise SystemExit(f"Tidak ada video di folder: {src}")
        return [str(v) for v in vids]
    if "://" not in src and not path.exists():
        raise SystemExit(f"Sumber tidak ditemukan: {src}")
    return [src]


def run_one(model, source, a, out_dir: Path):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Gagal membuka sumber: {source}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_live = isinstance(source, int) or "://" in str(source)
    name = f"webcam{source}" if isinstance(source, int) else (
        Path(source).stem if not is_live else "stream")

    writer = None
    if not a.no_save:
        out_path = out_dir / f"{name}_pred.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps / a.vid_stride, (w, h))
        print(f"Menyimpan ke: {out_path}")

    extra = {"half": True} if a.half else {}
    n, peak, sum_people, t0 = 0, 0, 0, time.time()
    idx = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % a.vid_stride:
                continue
            r = model.predict(frame, conf=a.conf, iou=a.iou, imgsz=a.imgsz, device=a.device,
                              max_det=a.max_det, verbose=False, **extra)[0]
            count = len(r.boxes)
            peak, sum_people, n = max(peak, count), sum_people + count, n + 1

            vis = r.plot(labels=not a.hide_labels, conf=not a.hide_conf, line_width=a.line_width)
            cv2.putText(vis, f"Orang: {count}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(vis, f"Orang: {count}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)
            if writer:
                writer.write(vis)
            if a.show:
                cv2.imshow("YOLO26 - Human Detection (q = keluar)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if n % 30 == 0:
                cur_fps = n / (time.time() - t0)
                prog = f"{idx + 1}/{total}" if total > 0 and not is_live else str(idx + 1)
                print(f"frame {prog} | {cur_fps:.1f} FPS | orang: {count}")
    finally:
        cap.release()
        if writer:
            writer.release()
        if a.show:
            cv2.destroyAllWindows()

    if n:
        print(f"[{name}] {n} frame diproses, rata-rata {sum_people / n:.1f} orang/frame, "
              f"puncak {peak}, {n / (time.time() - t0):.1f} FPS")


def main():
    a = parse_args()
    if a.weights == "auto":
        found = sorted((ROOT / "runs" / "train").rglob("best.pt"), key=lambda f: f.stat().st_mtime)
        if found:
            a.weights = str(found[-1])
            print(f"Memakai bobot: {a.weights}")
    if not Path(a.weights).exists():
        raise SystemExit(f"Bobot tidak ditemukan: {a.weights}\nTrain dulu (python train.py) atau isi --weights.")
    model = YOLO(a.weights)
    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in list_sources(a.source):
        run_one(model, s, a, out_dir)


if __name__ == "__main__":
    main()
