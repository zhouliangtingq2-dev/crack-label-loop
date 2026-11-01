#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a small synthetic "concrete wall with thin cracks" dataset so the whole
human-in-the-loop pipeline can be exercised end-to-end on a CPU in a few minutes.

No real photographs are used. Each image is built from:
  * a multi-scale noise texture with a mild illumination ramp (concrete-like background),
  * aggregate speckles and a few dark round blobs (unlabelled distractors: bolts, markers),
  * 1-3 random crack paths with slowly varying sub-pixel to few-pixel width (0.6-3.5 px),
    rendered at 4x super-sampling, darkened by a random depth factor, blurred by a
    Gaussian PSF and corrupted by sensor-like noise.

The "hand label" that a person would paint is written as a transparent PNG with the
crack colour #009BFF (0,155,255), exactly the format the real workflow uses.

Output layout (relative to --out):
  labelled/images/*.png      images that come with labels (the initial training set)
  labelled/labels/*.png      their labels (transparent PNG, #009BFF)
  pool/images/*.png          unlabelled images (what the model proposes labels for)
  pool/hidden_labels/*.png   ground truth for the pool, revealed only to simulate the human correction step
  test/images/*.png          held-out images for the final comparison
  test/hidden_labels/*.png   their ground truth
  truth_px.csv               per-image statistics (number of cracks, mean width, crack pixel ratio)

Usage:
  python scripts/00_make_demo_data.py --out demo_data --n-labelled 10 --n-pool 6 --n-test 4 --size 384 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CRACK_RGB = (0, 155, 255)  # #009BFF, same convention as the real workflow
SS = 4                      # super-sampling factor for crack rendering


def noise_field(rng: np.random.Generator, h: int, w: int, scale: int, amp: float) -> np.ndarray:
    """Smooth random field: low-res Gaussian noise up-sampled with bicubic interpolation."""
    hs, ws = max(2, h // scale), max(2, w // scale)
    small = rng.normal(0.0, 1.0, (hs, ws)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amp


def make_background(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    base = rng.uniform(120, 175)
    bg = np.full((h, w), base, np.float32)
    bg += noise_field(rng, h, w, 64, 9.0)    # large patches (formwork / moisture)
    bg += noise_field(rng, h, w, 16, 6.0)    # medium texture
    bg += noise_field(rng, h, w, 4, 4.0)     # fine grain
    # illumination ramp
    gx, gy = rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg *= 1.0 + gx * (xx / w - 0.5) + gy * (yy / h - 0.5)
    # aggregate speckles (small darker / lighter ellipses)
    n_speck = int(h * w / 900)
    for _ in range(n_speck):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        ax, ay = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        delta = rng.uniform(-14, 10)
        cv2.ellipse(bg, (cx, cy), (ax, ay), float(rng.uniform(0, 180)), 0, 360, float(base + delta), -1)
    bg = cv2.GaussianBlur(bg, (0, 0), 0.6)
    return bg


def add_distractors(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    """Dark round blobs that are NOT cracks (bolts, markers, holes). They are not labelled."""
    h, w = img.shape
    for _ in range(int(rng.integers(0, 4))):
        cx, cy = int(rng.integers(10, w - 10)), int(rng.integers(10, h - 10))
        r = int(rng.integers(3, 9))
        dark = rng.uniform(0.15, 0.45)
        cv2.circle(img, (cx, cy), r, float(img[cy, cx] * dark), -1, lineType=cv2.LINE_AA)
    return img


def crack_path(rng: np.random.Generator, h: int, w: int) -> list[tuple[float, float]]:
    """Random walk with slowly changing heading; starts on a random image edge."""
    side = int(rng.integers(0, 4))
    if side == 0:
        x, y, heading = rng.uniform(0, w), 0.0, rng.uniform(math.pi * 0.2, math.pi * 0.8)
    elif side == 1:
        x, y, heading = rng.uniform(0, w), float(h), -rng.uniform(math.pi * 0.2, math.pi * 0.8)
    elif side == 2:
        x, y, heading = 0.0, rng.uniform(0, h), rng.uniform(-math.pi * 0.3, math.pi * 0.3)
    else:
        x, y, heading = float(w), rng.uniform(0, h), math.pi + rng.uniform(-math.pi * 0.3, math.pi * 0.3)
    pts = [(x, y)]
    step = 5.0
    wobble = rng.uniform(0.03, 0.10)                 # per-step heading jitter (rad)
    curv_amp, curv_per = rng.uniform(0.0, 0.035), rng.uniform(25, 70)  # slow bending
    for i in range(400):
        heading += rng.normal(0.0, wobble) + curv_amp * math.sin(i / curv_per)
        x += step * math.cos(heading)
        y += step * math.sin(heading)
        pts.append((x, y))
        if x < -5 or x > w + 5 or y < -5 or y > h + 5:
            break
    return pts


def render_cracks(rng: np.random.Generator, h: int, w: int, n_cracks: int):
    """Return (coverage alpha in [0,1] at full resolution, list of (mean_width_px))."""
    canvas = np.zeros((h * SS, w * SS), np.float32)
    widths = []
    for _ in range(n_cracks):
        pts = crack_path(rng, h, w)
        w0 = float(rng.choice([rng.uniform(0.5, 1.2), rng.uniform(1.0, 2.5), rng.uniform(2.0, 3.5)]))  # mean width in px, sub-pixel allowed
        seg_w = []
        for i in range(len(pts) - 1):
            wi = max(0.3, w0 * (1.0 + 0.35 * math.sin(i / rng.uniform(15, 40)) + rng.normal(0, 0.05)))
            seg_w.append(wi)
            p0 = (int(round(pts[i][0] * SS)), int(round(pts[i][1] * SS)))
            p1 = (int(round(pts[i + 1][0] * SS)), int(round(pts[i + 1][1] * SS)))
            cv2.line(canvas, p0, p1, 1.0, max(1, int(round(wi * SS))), lineType=cv2.LINE_8)
        widths.append(float(np.mean(seg_w)))
        # occasional short branch
        if rng.random() < 0.4 and len(pts) > 40:
            j = int(rng.integers(10, len(pts) - 10))
            bx, by = pts[j]
            ang = rng.uniform(0, 2 * math.pi)
            for k in range(int(rng.integers(5, 25))):
                nx, ny = bx + 5 * math.cos(ang), by + 5 * math.sin(ang)
                cv2.line(canvas, (int(bx * SS), int(by * SS)), (int(nx * SS), int(ny * SS)), 1.0, max(1, int(round(w0 * 0.6 * SS))))
                bx, by = nx, ny
                ang += rng.normal(0, 0.25)
    alpha = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)  # fractional pixel coverage
    return np.clip(alpha, 0, 1), widths


def make_image(rng: np.random.Generator, size: int):
    h = w = size
    bg = make_background(rng, h, w)
    bg = add_distractors(rng, bg)
    n_cracks = int(rng.integers(1, 4))
    alpha, widths = render_cracks(rng, h, w, n_cracks)
    depth = rng.uniform(0.55, 0.95)                     # how dark the crack core is (D)
    img = bg * (1.0 - depth * alpha)
    sigma = rng.uniform(0.6, 1.2)                       # PSF blur in px
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    img += rng.normal(0.0, rng.uniform(1.5, 4.0), img.shape)  # sensor noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    # what a careful person would paint: pixels with >= 30 % crack coverage
    label = alpha >= 0.30
    # guarantee at least a 1-px line where the crack is very thin
    thin = (alpha >= 0.10) & ~label
    label |= thin & (cv2.dilate(label.astype(np.uint8), np.ones((3, 3), np.uint8)) == 0) & (alpha >= 0.18)
    rgb = np.stack([img, img, img], axis=2)
    # slight colour cast so the image is a genuine RGB photo-like input
    cast = rng.uniform(0.96, 1.04, 3)
    rgb = np.clip(rgb.astype(np.float32) * cast, 0, 255).astype(np.uint8)
    stats = dict(n_cracks=n_cracks, mean_width_px=float(np.mean(widths)), depth=depth, psf_sigma_px=sigma,
                 crack_px_ratio=float(label.mean()))
    return rgb, label, stats


def label_png(label: np.ndarray) -> Image.Image:
    rgba = np.zeros((label.shape[0], label.shape[1], 4), np.uint8)
    rgba[label] = (*CRACK_RGB, 255)
    return Image.fromarray(rgba, mode="RGBA")


def write_set(rng, out: Path, prefix: str, n: int, size: int, label_dir_name: str, rows: list):
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / label_dir_name).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        rgb, label, stats = make_image(rng, size)
        name = f"{prefix}_{i:03d}"
        Image.fromarray(rgb).save(out / "images" / f"{name}.png")
        label_png(label).save(out / label_dir_name / f"{name}.png")
        rows.append(dict(set=out.name, name=name, **stats))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("demo_data"))
    ap.add_argument("--n-labelled", type=int, default=10)
    ap.add_argument("--n-pool", type=int, default=6)
    ap.add_argument("--n-test", type=int, default=4)
    ap.add_argument("--size", type=int, default=384, help="image side length in px")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    write_set(rng, args.out / "labelled", "lab", args.n_labelled, args.size, "labels", rows)
    write_set(rng, args.out / "pool", "pool", args.n_pool, args.size, "hidden_labels", rows)
    write_set(rng, args.out / "test", "test", args.n_test, args.size, "hidden_labels", rows)
    with (args.out / "truth_px.csv").open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"demo data written to {args.out}: labelled={args.n_labelled} pool={args.n_pool} test={args.n_test} size={args.size}")


if __name__ == "__main__":
    main()
