#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run prediction + evaluation for several probability thresholds and tabulate the metrics.

Usage:
  python scripts/07_threshold_sweep.py --checkpoint runs/r1/best_model.pth --images data/processed/images --masks data/processed/masks_id --split data/splits/val.txt --out runs/sweep --thresholds 0.30 0.40 0.50 0.60
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read_split(split_path: Path):
    names = []
    for line in split_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            names.append(Path(s).stem)
    return names


def find_image_by_stem(folder: Path, stem: str):
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    matches = list(folder.glob(stem + ".*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find image for stem={stem} in {folder}")


def make_split_image_folder(images_dir: Path, split_path: Path, out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = read_split(split_path)
    for stem in names:
        src = find_image_by_stem(images_dir, stem)
        shutil.copy2(src, out_dir / f"{stem}{src.suffix.lower()}")

    return names


def threshold_tag(th):
    return f"t{int(round(float(th) * 100)):03d}"


def run_cmd(cmd):
    print("\n[RUN]")
    print(" ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def read_summary(summary_csv: Path):
    d = {}
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                d[row[0]] = float(row[1])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thresholds", nargs="+", required=True)
    ap.add_argument("--patch-size", type=int, default=None, help="default: the patch size stored in the checkpoint")
    ap.add_argument("--overlap", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--min-area", type=int, default=1)
    ap.add_argument("--erode-pixels", type=int, default=0)
    ap.add_argument("--erode-kernel", default="cross")
    ap.add_argument("--skeleton-dilate", type=int, default=0)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    split_path = Path(args.split)
    out_dir = Path(args.out)

    out_dir.mkdir(parents=True, exist_ok=True)

    input_dir = out_dir / "_split_images"
    names = make_split_image_folder(images_dir, split_path, input_dir)

    rows = []

    for th in args.thresholds:
        tag = threshold_tag(th)
        pred_dir = out_dir / f"pred_{tag}"
        eval_csv = out_dir / f"eval_{tag}.csv"

        if pred_dir.exists():
            shutil.rmtree(pred_dir)

        cmd = [
            sys.executable,
            str(HERE / "05_predict.py"),
            "--checkpoint", str(checkpoint),
            "--images", str(input_dir),
            "--out", str(pred_dir),
            "--overlap", str(args.overlap),
            "--batch-size", str(args.batch_size),
            "--threshold", str(th),
            "--min-area", str(args.min_area),
            "--erode-pixels", str(args.erode_pixels),
            "--erode-kernel", args.erode_kernel,
            "--skeleton-dilate", str(args.skeleton_dilate),
        ]
        if args.patch_size is not None:
            cmd += ["--patch-size", str(args.patch_size)]
        run_cmd(cmd)

        run_cmd([
            sys.executable,
            str(HERE / "06_evaluate.py"),
            "--pred", str(pred_dir / "affinity_png"),
            "--gt", str(masks_dir),
            "--split", str(split_path),
            "--out", str(eval_csv),
        ])

        summary = read_summary(eval_csv.with_name(eval_csv.stem + "_summary.csv"))

        rows.append({
            "threshold": float(th),
            "tag": tag,
            "micro_dice": summary.get("micro_dice", 0.0),
            "micro_iou": summary.get("micro_iou", 0.0),
            "micro_precision": summary.get("micro_precision", 0.0),
            "micro_recall": summary.get("micro_recall", 0.0),
            "mean_dice": summary.get("mean_dice", 0.0),
            "mean_iou": summary.get("mean_iou", 0.0),
            "mean_precision": summary.get("mean_precision", 0.0),
            "mean_recall": summary.get("mean_recall", 0.0),
            "n_images": len(names),
            "prediction_folder": str(pred_dir),
        })

    summary_out = out_dir / "threshold_sweep_summary.csv"
    with summary_out.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "threshold",
            "tag",
            "micro_dice",
            "micro_iou",
            "micro_precision",
            "micro_recall",
            "mean_dice",
            "mean_iou",
            "mean_precision",
            "mean_recall",
            "n_images",
            "prediction_folder",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda r: r["micro_dice"])

    print("\n========== THRESHOLD SWEEP DONE ==========")
    print("Summary:", summary_out)
    print("Best by micro_dice:")
    print(best)


if __name__ == "__main__":
    main()