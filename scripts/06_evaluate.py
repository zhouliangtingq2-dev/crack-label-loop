"""Pixel-level evaluation of predicted label PNGs (#009BFF on transparent) against masks_id ground truth.
Pixels with mask value 255 (ignore / spalling) are excluded. Writes per-image CSV and a *_summary.csv
with micro (pixel-pooled) and mean (per-image) IoU / Dice / precision / recall.

Usage:
  python scripts/06_evaluate.py --pred runs/pred/affinity_png --gt data/processed/masks_id --split data/splits/test.txt --out runs/eval/test.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def read_split(path):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    names = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            names.append(Path(s).stem)
    return names


def find_by_stem(folder, stem):
    folder = Path(folder)
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    matches = list(folder.glob(stem + ".*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find {stem} in {folder}")


def load_gt(mask_path):
    """
    masks_id:
      0   = background
      1   = crack
      255 = ignore / spalling
    """
    m = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
    gt = m == 1
    valid = m != 255
    return gt, valid


def load_pred_affinity(pred_path):
    """
    Affinity PNG:
      crack = #009BFF with alpha
      background = transparent
    """
    im = Image.open(pred_path)
    rgba = np.array(im.convert("RGBA"), dtype=np.uint8)

    r = rgba[:, :, 0]
    g = rgba[:, :, 1]
    b = rgba[:, :, 2]
    a = rgba[:, :, 3]

    blue = (r < 80) & (g > 80) & (b > 120)

    # transparent PNG: also require alpha > 0; plain RGB PNG: colour test only
    has_alpha = "A" in im.getbands()
    if has_alpha:
        return (a > 0) & blue
    else:
        return blue


def resize_nearest_bool(mask, target_shape):
    h, w = target_shape
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((w, h), resample=Image.Resampling.NEAREST)
    return np.array(img) > 0


def calc_metrics(pred, gt, valid):
    pred = pred & valid
    gt = gt & valid

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    iou = tp / (tp + fp + fn + 1e-8)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return int(tp), int(fp), int(fn), float(iou), float(dice), float(precision), float(recall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="prediction affinity_png folder")
    ap.add_argument("--gt", required=True, help="ground truth masks_id folder")
    ap.add_argument("--split", default=None, help="optional split txt")
    ap.add_argument("--out", required=True, help="output csv")
    args = ap.parse_args()

    pred_dir = Path(args.pred)
    gt_dir = Path(args.gt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    names = read_split(args.split)
    if names is None:
        names = sorted([p.stem for p in gt_dir.glob("*.png")])

    rows = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for stem in names:
        gt_path = find_by_stem(gt_dir, stem)
        pred_path = find_by_stem(pred_dir, stem)

        gt, valid = load_gt(gt_path)
        pred = load_pred_affinity(pred_path)

        if pred.shape != gt.shape:
            pred = resize_nearest_bool(pred, gt.shape)

        tp, fp, fn, iou, dice, precision, recall = calc_metrics(pred, gt, valid)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        rows.append({
            "name": stem,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
        })

    micro_iou = total_tp / (total_tp + total_fp + total_fn + 1e-8)
    micro_dice = 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8)
    micro_precision = total_tp / (total_tp + total_fp + 1e-8)
    micro_recall = total_tp / (total_tp + total_fn + 1e-8)

    mean_iou = float(np.mean([r["iou"] for r in rows]))
    mean_dice = float(np.mean([r["dice"] for r in rows]))
    mean_precision = float(np.mean([r["precision"] for r in rows]))
    mean_recall = float(np.mean([r["recall"] for r in rows]))

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "tp", "fp", "fn", "iou", "dice", "precision", "recall"]
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["micro_iou", micro_iou])
        writer.writerow(["micro_dice", micro_dice])
        writer.writerow(["micro_precision", micro_precision])
        writer.writerow(["micro_recall", micro_recall])
        writer.writerow(["mean_iou", mean_iou])
        writer.writerow(["mean_dice", mean_dice])
        writer.writerow(["mean_precision", mean_precision])
        writer.writerow(["mean_recall", mean_recall])
        writer.writerow(["n_images", len(rows)])

    print("Saved:", out_path)
    print("Saved:", summary_path)
    print(f"micro_iou       = {micro_iou:.4f}")
    print(f"micro_dice      = {micro_dice:.4f}")
    print(f"micro_precision = {micro_precision:.4f}")
    print(f"micro_recall    = {micro_recall:.4f}")
    print(f"mean_iou        = {mean_iou:.4f}")
    print(f"mean_dice       = {mean_dice:.4f}")
    print(f"mean_precision  = {mean_precision:.4f}")
    print(f"mean_recall     = {mean_recall:.4f}")


if __name__ == "__main__":
    main()