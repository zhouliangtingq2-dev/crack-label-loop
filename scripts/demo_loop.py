#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end demonstration of the human-in-the-loop labelling loop on synthetic data (CPU-friendly).

  round 1 : labelled set (N images)      -> prepare -> split -> augment -> train -> propose labels for the pool
  "human" : the pool proposals are corrected (simulated here by revealing the hidden ground truth)
            and the corrected labels are added to the labelled set
  round 2 : labelled set (N + pool)      -> retrain -> evaluate on the held-out test set

Everything is done by calling the numbered scripts exactly as a user would, so the demo is also a
smoke test of the whole pipeline.

Usage (defaults: ~5 min on a laptop CPU, seconds on a GPU):
  python scripts/demo_loop.py --root demo_runs
  python scripts/demo_loop.py --root demo_runs --no-pretrained          # offline / CI
  python scripts/demo_loop.py --root demo_runs --epochs 8 --steps 80    # a bit better quality
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
CRACK_RGB = (0, 155, 255)


def run(cmd: list, log: list):
    cmd = [str(c) for c in cmd]
    print("\n[RUN] " + " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    log.append((Path(cmd[1]).name, time.time() - t0))


def read_summary(p: Path) -> dict:
    d = {}
    with p.open(encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) >= 2:
                d[row[0]] = float(row[1])
    return d


def pipeline_round(tag: str, root: Path, labelled: Path, args, log: list) -> Path:
    """prepare -> split -> augment -> train for one round; returns the run folder."""
    r = root / tag
    if r.exists():
        shutil.rmtree(r)
    py = sys.executable
    run([py, HERE / "01_prepare_dataset.py", "--images", labelled / "images", "--labels", labelled / "labels", "--out", r / "processed"], log)
    run([py, HERE / "02_make_splits.py", "--data", r / "processed", "--out", r / "splits", "--train", "0.8", "--val", "0.2", "--test", "0", "--seed", args.seed], log)
    run([py, HERE / "03_augment_dataset.py", "--input", r / "processed", "--splits", r / "splits", "--out", r / "processed_aug", "--out-splits", r / "splits_aug"], log)
    train_cmd = [py, HERE / "04_train.py", "--data", r / "processed_aug", "--splits", r / "splits_aug", "--out", r / "model",
                 "--model", args.model, "--encoder", args.encoder, "--patch-size", args.patch, "--batch-size", args.batch,
                 "--epochs", args.epochs, "--steps-per-epoch", args.steps, "--lr", "0.0005", "--num-workers", args.workers,
                 "--threshold", args.threshold, "--pos-weight", "8", "--seed", args.seed]
    if args.no_pretrained:
        train_cmd.append("--no-pretrained")
    run(train_cmd, log)
    run([py, HERE / "08_plot_history.py", "--history", r / "model" / "history.csv", "--out", r / "model" / "history.png"], log)
    return r


def predict_and_eval(r: Path, name: str, images: Path, truth_masks: Path, args, log: list) -> dict:
    py = sys.executable
    pred = r / f"pred_{name}"
    run([py, HERE / "05_predict.py", "--checkpoint", r / "model" / "best_model.pth", "--images", images, "--out", pred,
         "--overlap", max(16, args.patch // 2), "--batch-size", args.batch, "--threshold", args.threshold, "--min-area", "5",
         "--erode-pixels", args.erode, "--erode-kernel", "cross", "--no-amp"], log)
    run([py, HERE / "06_evaluate.py", "--pred", pred / "affinity_png", "--gt", truth_masks, "--out", r / f"eval_{name}.csv"], log)
    return read_summary(r / f"eval_{name}_summary.csv")


def truth_masks(root: Path, name: str, images: Path, labels: Path, log: list) -> Path:
    """Convert hidden labels to masks_id with the same script used for training data."""
    out = root / f"truth_{name}"
    if out.exists():
        shutil.rmtree(out)
    run([sys.executable, HERE / "01_prepare_dataset.py", "--images", images, "--labels", labels, "--out", out], log)
    return out / "masks_id"


def overlay(img: np.ndarray, mask: np.ndarray, color=CRACK_RGB, alpha=0.9) -> np.ndarray:
    out = img.astype(np.float32).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def load_label_mask(p: Path) -> np.ndarray:
    a = np.array(Image.open(p).convert("RGBA"))
    return (a[..., 3] > 0) & (a[..., 0] < 80) & (a[..., 1] > 80) & (a[..., 2] > 120)


def montage(root: Path, r1: Path, r2: Path, test_images: Path, test_labels: Path, out_png: Path, n: int = 2):
    """input | round-1 proposal | round-2 result | ground truth, for the first n test images."""
    rows = []
    for p in sorted(test_images.iterdir())[:n]:
        img = np.array(Image.open(p).convert("RGB"))
        gt = load_label_mask(test_labels / p.name)
        m1 = load_label_mask(r1 / "pred_test" / "affinity_png" / p.name)
        m2 = load_label_mask(r2 / "pred_test" / "affinity_png" / p.name)
        rows.append(np.concatenate([img, overlay(img, m1), overlay(img, m2), overlay(img, gt, (255, 170, 0))], axis=1))
    sheet = np.concatenate(rows, axis=0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet).save(out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("demo_runs"))
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--n-labelled", type=int, default=6)
    ap.add_argument("--n-pool", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=4)
    ap.add_argument("--model", default="unet")
    ap.add_argument("--encoder", default="resnet18")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=50, help="training steps per epoch")
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--erode", type=int, default=0, help="protected-erosion pixels at prediction time (use 1-2 on real photos where the model over-paints)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--keep-data", action="store_true", help="reuse --root/data if it already exists")
    args = ap.parse_args()

    root = args.root
    data = root / "data"
    log: list = []
    t_all = time.time()
    if not (args.keep_data and data.exists()):
        if data.exists():
            shutil.rmtree(data)
        run([sys.executable, HERE / "00_make_demo_data.py", "--out", data, "--n-labelled", args.n_labelled, "--n-pool", args.n_pool,
             "--n-test", args.n_test, "--size", args.size, "--seed", args.seed], log)
    labelled = data / "labelled"
    pool, test = data / "pool", data / "test"
    tm_pool = truth_masks(root, "pool", pool / "images", pool / "hidden_labels", log)
    tm_test = truth_masks(root, "test", test / "images", test / "hidden_labels", log)

    # ---------------- round 1: train on the initial labelled set, propose labels for the pool
    r1 = pipeline_round("round1", root, labelled, args, log)
    s1_pool = predict_and_eval(r1, "pool", pool / "images", tm_pool, args, log)
    s1_test = predict_and_eval(r1, "test", test / "images", tm_test, args, log)

    # ---------------- "human correction": in real use you open round1/pred_pool/affinity_png/*.png in an image
    # editor, fix the proposal, and save it next to the other labels. Here the corrected labels are the hidden
    # truth, i.e. a *perfect* correction. The enlarged set is written to its own folder (data/ is never modified).
    labelled2 = root / "labelled_round2"
    if labelled2.exists():
        shutil.rmtree(labelled2)
    shutil.copytree(labelled, labelled2)
    for p in sorted((pool / "images").iterdir()):
        shutil.copy2(p, labelled2 / "images" / p.name)
        shutil.copy2(pool / "hidden_labels" / p.name, labelled2 / "labels" / p.name)

    # ---------------- round 2: retrain on labelled + corrected pool, evaluate on the held-out test set
    r2 = pipeline_round("round2", root, labelled2, args, log)
    s2_test = predict_and_eval(r2, "test", test / "images", tm_test, args, log)

    # ---------------- report
    rep = root / "report"
    rep.mkdir(exist_ok=True)
    montage(root, r1, r2, test / "images", test / "hidden_labels", rep / "demo_result.png")
    shutil.copy2(r2 / "model" / "history_metrics.png", rep / "round2_history_metrics.png")
    keys = ["micro_iou", "micro_dice", "micro_precision", "micro_recall"]
    lines = ["# Demo loop report", "",
             f"synthetic data: labelled={args.n_labelled} (+{args.n_pool} after correction), test={args.n_test}, size={args.size}px; "
             f"model={args.model}/{args.encoder}, patch={args.patch}, epochs={args.epochs}x{args.steps} steps, threshold={args.threshold}, "
             f"pretrained={'no' if args.no_pretrained else 'ImageNet'}", "",
             "| set | round | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys),
             "| pool (proposals the human corrects) | 1 | " + " | ".join(f"{s1_pool[k]:.3f}" for k in keys) + " |",
             "| test (held out) | 1 | " + " | ".join(f"{s1_test[k]:.3f}" for k in keys) + " |",
             "| test (held out) | 2 | " + " | ".join(f"{s2_test[k]:.3f}" for k in keys) + " |", "",
             "![result](demo_result.png)", "",
             "columns: input | round-1 proposal | round-2 result | ground truth (orange)", "",
             "note: the 'human correction' between the rounds is simulated by revealing the hidden ground truth of the pool,",
             "so the round-2 gain shows the effect of adding correctly labelled images, not the effort saved by a human.", "",
             "## timing", "", "| step | seconds |", "|---|---|"] + [f"| {n} | {t:.1f} |" for n, t in log] + \
            ["", f"total {time.time() - t_all:.0f} s"]
    (rep / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:12]))
    print("\nreport:", rep / "report.md")


if __name__ == "__main__":
    main()
