"""Split the processed dataset into train / val / test lists (by image stem).

Usage:
  python scripts/02_make_splits.py --data data/processed --out data/splits --train 0.7 --val 0.15 --test 0.15 --seed 42
A test ratio of 0 is allowed (all images go to train/val).
"""
import argparse
import random
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, help="processed folder")
    ap.add_argument("--out", required=True, type=Path, help="splits folder")
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    masks = sorted([p for p in (args.data / "masks_id").iterdir() if p.suffix.lower() in IMG_EXTS])
    stems = [p.stem for p in masks]
    random.Random(args.seed).shuffle(stems)
    n = len(stems)
    if n < 3:
        raise SystemExit("Need at least 3 images to make train/val/test split.")
    n_train = max(1, int(round(n * args.train)))
    n_val = max(1, int(round(n * args.val)))
    if args.test <= 0:
        n_train = max(1, min(n_train, n - 1))
        n_val = n - n_train
    elif n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train = stems[:n_train]
    val = stems[n_train:n_train + n_val]
    test = stems[n_train + n_val:]

    args.out.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        (args.out / f"{name}.txt").write_text("\n".join(split) + "\n", encoding="utf-8")
        print(name, len(split))


if __name__ == "__main__":
    main()
