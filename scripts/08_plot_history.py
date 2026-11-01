"""Plot loss and validation metrics from a training history.csv.

Usage:
  python scripts/08_plot_history.py --history runs/r1/history.csv --out runs/r1/history.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_history(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def get_col(rows, candidates):
    keys = rows[0].keys()
    for c in candidates:
        if c in keys:
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[c]))
                except Exception:
                    vals.append(None)
            return c, vals
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="Path to history.csv")
    ap.add_argument("--out", required=True, help="Output PNG path")
    args = ap.parse_args()

    history_path = Path(args.history)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_history(history_path)

    epoch_name, epochs = get_col(rows, ["epoch", "epochs"])
    if epochs is None:
        epochs = list(range(1, len(rows) + 1))

    # tolerate different column names
    train_name, train_loss = get_col(rows, ["train_loss", "train", "loss_train"])
    val_name, val_loss = get_col(rows, ["val_loss", "val", "loss_val"])
    iou_name, val_iou = get_col(rows, ["val_iou", "iou"])
    dice_name, val_dice = get_col(rows, ["val_dice", "dice"])
    score_name, score = get_col(rows, ["score", "val_score"])

    # Figure 1: loss
    plt.figure(figsize=(8, 5))
    plotted = False
    if train_loss is not None:
        plt.plot(epochs, train_loss, label=train_name)
        plotted = True
    if val_loss is not None:
        plt.plot(epochs, val_loss, label=val_name)
        plotted = True
    if not plotted:
        raise ValueError("Could not find train/val loss columns in history.csv")

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training curve: loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    loss_out = out_path.with_name(out_path.stem + "_loss.png")
    plt.savefig(loss_out, dpi=200)
    plt.close()

    # Figure 2: metrics
    plt.figure(figsize=(8, 5))
    plotted = False
    if val_iou is not None:
        plt.plot(epochs, val_iou, label=iou_name)
        plotted = True
    if val_dice is not None:
        plt.plot(epochs, val_dice, label=dice_name)
        plotted = True
    if score is not None:
        plt.plot(epochs, score, label=score_name)
        plotted = True

    if plotted:
        plt.xlabel("epoch")
        plt.ylabel("metric")
        plt.title("Validation metrics")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        metric_out = out_path.with_name(out_path.stem + "_metrics.png")
        plt.savefig(metric_out, dpi=200)
        plt.close()
        print("Saved:", metric_out)

    print("Saved:", loss_out)


if __name__ == "__main__":
    main()