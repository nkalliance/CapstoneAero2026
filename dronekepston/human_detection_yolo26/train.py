"""Training YOLO26 untuk deteksi manusia dari drone.

Contoh:
    python train.py                                  # default: yolo26n, 100 epoch
    python train.py --model yolo26s.pt --imgsz 960 --batch 8
    python train.py --device cpu --epochs 5          # uji cepat tanpa GPU
    python train.py --resume                         # lanjutkan run terakhir
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent


def find_latest(pattern: str):
    """Cari file terbaru di runs/train/*/weights/ (mis. last.pt)."""
    files = sorted((ROOT / "runs" / "train").rglob(pattern), key=lambda f: f.stat().st_mtime)
    return files[-1] if files else None


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO26 - Human Detection Drone")
    p.add_argument("--data", default=str(ROOT / "dataset" / "data.yaml"), help="path data.yaml")
    p.add_argument("--model", default="yolo26n.pt",
                   help="bobot awal: yolo26n/s/m/l/x.pt (pretrained) atau yolo26n.yaml (dari nol)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640,
                   help="ukuran gambar. Objek manusia dari drone kecil -> coba 960 atau 1280")
    p.add_argument("--batch", type=int, default=16, help="-1 = otomatis sesuai VRAM")
    p.add_argument("--device", default=None, help="'0', '0,1', 'cpu', 'mps' (kosong = otomatis)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=30, help="early stopping (epoch tanpa perbaikan)")
    p.add_argument("--project", default=str(ROOT / "runs" / "train"))
    p.add_argument("--name", default="exp")
    p.add_argument("--cache", default="False", help="True/ram, disk, atau False")
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="lanjutkan training. Tanpa nilai = otomatis pakai last.pt terbaru; "
                        "atau isi path last.pt")
    p.add_argument("--fraction", type=float, default=1.0, help="pakai sebagian dataset (untuk uji cepat)")
    p.add_argument("--no-val-test", action="store_true", help="lewati evaluasi di split test")
    return p.parse_args()


def main():
    a = parse_args()

    if a.resume:
        ckpt = find_latest("last.pt") if a.resume == "auto" else Path(a.resume)
        if ckpt is None or not Path(ckpt).exists():
            raise SystemExit(f"last.pt tidak ditemukan ({a.resume}). Cek folder runs/train/*/weights/ "
                             "atau isi path lengkapnya, mis. --resume C:/path/ke/last.pt")
        print(f"Melanjutkan dari: {ckpt}")
        YOLO(str(ckpt)).train(resume=True)
        return

    model = YOLO(a.model)
    cache = {"true": True, "false": False}.get(a.cache.lower(), a.cache)

    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        workers=a.workers,
        patience=a.patience,
        project=a.project,
        name=a.name,
        cache=cache,
        fraction=a.fraction,
        exist_ok=False,
        # augmentasi yang cocok untuk citra udara (drone): flip vertikal juga masuk akal
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        scale=0.5,
        degrees=10,
    )

    # Evaluasi akhir dengan bobot terbaik
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nBobot terbaik: {best}")
    best_model = YOLO(str(best))
    for split in ["val"] + ([] if a.no_val_test else ["test"]):
        try:
            m = best_model.val(data=a.data, split=split, imgsz=a.imgsz, device=a.device,
                                    project=str(model.trainer.save_dir), name=f"eval_{split}", exist_ok=True)
            print(f"[{split}] mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f} "
                  f"P={m.box.mp:.3f} R={m.box.mr:.3f}")
        except Exception as e:  # split tidak ada, dsb.
            print(f"[{split}] dilewati: {e}")


if __name__ == "__main__":
    main()
