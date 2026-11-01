#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline augmentation of the *training* split only: 90-degree rotations and flips (no interpolation,
so thin masks stay exact). val/test are copied unchanged.

Usage:
  python scripts/03_augment_dataset.py --input data/processed --splits data/splits --out data/processed_aug --out-splits data/splits_aug
"""
from pathlib import Path
import argparse
import shutil
import csv

import numpy as np
from PIL import Image


def read_split(path):
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                names.append(s)
    return names


def find_file(folder, name):
    folder = Path(folder)
    p = folder / name
    if p.exists():
        return p

    stem = Path(name).stem
    for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p

    matches = list(folder.glob(stem + ".*"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Cannot find {name} in {folder}")


def save_rgb(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path)


def save_mask(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path)


def color_preview(mask):
    out = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    out[mask == 1] = (0, 155, 255)       # crack #009BFF
    out[mask == 255] = (255, 255, 0)     # ignore / spalling
    return out


def augment_pairs(image, mask):
    """Geometric transforms applied identically to image and mask; only 90-degree rotations and flips, so no interpolation."""
    variants = []

    variants.append(("orig", image, mask))
    variants.append(("r90", np.rot90(image, 1), np.rot90(mask, 1)))
    variants.append(("r180", np.rot90(image, 2), np.rot90(mask, 2)))
    variants.append(("r270", np.rot90(image, 3), np.rot90(mask, 3)))
    variants.append(("fliplr", np.fliplr(image), np.fliplr(mask)))
    variants.append(("flipud", np.flipud(image), np.flipud(mask)))

    return variants


def copy_one(input_dir, out_dir, name, augment=False):
    img_path = find_file(input_dir / "images", name)
    mask_path = find_file(input_dir / "masks_id", name)

    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)

    stem = Path(img_path).stem
    saved_names = []

    if augment:
        variants = augment_pairs(img, mask)
    else:
        variants = [("orig", img, mask)]

    for suffix, img_aug, mask_aug in variants:
        new_name = f"{stem}__{suffix}.png"

        save_rgb(img_aug, out_dir / "images" / new_name)
        save_mask(mask_aug, out_dir / "masks_id" / new_name)
        save_rgb(color_preview(mask_aug), out_dir / "preview_color" / new_name)

        saved_names.append(new_name)

    return saved_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="processed dataset folder (images/, masks_id/)")
    ap.add_argument("--splits", required=True, help="splits folder (train.txt, val.txt, test.txt)")
    ap.add_argument("--out", required=True, help="output augmented dataset folder")
    ap.add_argument("--out-splits", required=True, help="output splits folder")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    input_dir = Path(args.input)
    splits_dir = Path(args.splits)
    out_dir = Path(args.out)
    out_splits = Path(args.out_splits)

    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    if args.overwrite and out_splits.exists():
        shutil.rmtree(out_splits)

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks_id").mkdir(parents=True, exist_ok=True)
    (out_dir / "preview_color").mkdir(parents=True, exist_ok=True)
    out_splits.mkdir(parents=True, exist_ok=True)

    train = read_split(splits_dir / "train.txt")
    val = read_split(splits_dir / "val.txt")
    test = read_split(splits_dir / "test.txt")

    metadata = []

    split_outputs = {
        "train": [],
        "val": [],
        "test": [],
    }

    # augment the training split only
    for name in train:
        saved = copy_one(input_dir, out_dir, name, augment=True)
        split_outputs["train"].extend(saved)
        for s in saved:
            metadata.append(["train", name, s])

    # val/test: originals only
    for name in val:
        saved = copy_one(input_dir, out_dir, name, augment=False)
        split_outputs["val"].extend(saved)
        for s in saved:
            metadata.append(["val", name, s])

    for name in test:
        saved = copy_one(input_dir, out_dir, name, augment=False)
        split_outputs["test"].extend(saved)
        for s in saved:
            metadata.append(["test", name, s])

    for split_name, names in split_outputs.items():
        with open(out_splits / f"{split_name}.txt", "w", encoding="utf-8") as f:
            for n in names:
                f.write(n + "\n")

    with open(out_dir / "augmentation_metadata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "source_name", "augmented_name"])
        writer.writerows(metadata)

    print("Done augmented dataset.")
    print("Original train:", len(train))
    print("Augmented train:", len(split_outputs["train"]))
    print("Val:", len(split_outputs["val"]))
    print("Test:", len(split_outputs["test"]))
    print("Output:", out_dir)
    print("Output splits:", out_splits)


if __name__ == "__main__":
    main()