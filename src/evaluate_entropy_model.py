"""Evaluate the SPATCH entropy-selected nnU-Net model on the five target ROIs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import binary_metrics, load_rois, save_binary_mask


def normalize(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    return np.clip(
        (image.astype(np.float32) - low) / max(float(high - low), 1.0), 0, 1
    )


def boundary(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return mask & ~interior


def case_name(roi_name: str) -> str:
    coordinate = roi_name.split("-", 1)[1].removesuffix("_256_256")
    return f"nuclei_{coordinate}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.571)
    parser.add_argument("--held-out", default="nuclei_18128_21684")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rois = load_rois(args.annotations)
    rows: list[dict[str, object]] = []
    predictions: list[np.ndarray] = []

    for roi in rois:
        case = case_name(roi.name)
        probability = np.squeeze(
            np.load(args.probabilities / f"{case}.npz")["probabilities"][1]
        )
        prediction = probability >= args.threshold
        predictions.append(prediction)
        save_binary_mask(args.predictions / f"{roi.name}_mask.tif", prediction)
        metric = binary_metrics(prediction, roi.mask)
        rows.append(
            {
                "method": "nnunet_entropy_spatch_fold0",
                "roi": roi.name,
                "split": "held-out" if case == args.held_out else "train",
                "threshold": args.threshold,
                **metric,
            }
        )

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(len(rois), 3, figsize=(7.4, 12.0))
    for index, (roi, prediction, row) in enumerate(zip(rois, predictions, rows)):
        display = normalize(roi.image)
        axes[index, 0].imshow(display, cmap="gray", vmin=0, vmax=1)
        axes[index, 1].imshow(roi.mask, cmap="gray", vmin=0, vmax=1)
        axes[index, 2].imshow(display, cmap="gray", vmin=0, vmax=1)
        axes[index, 2].contour(
            boundary(prediction).astype(np.uint8),
            levels=[0.5],
            colors=["#00A878"],
            linewidths=0.8,
        )
        axes[index, 0].set_ylabel(
            f"ROI {index + 1}\n{row['split']}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
        )
        axes[index, 2].text(
            0.5,
            -0.045,
            f"D={float(row['Dice']):.3f}  FNR={100 * float(row['FNR']):.1f}%  "
            f"FDR={100 * float(row['FDR']):.1f}%",
            transform=axes[index, 2].transAxes,
            ha="center",
            va="top",
            fontsize=7,
        )
        for axis in axes[index]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
    for axis, title in zip(
        axes[0], ("DAPI", "Manual mask", "Entropy-selected nnU-Net"), strict=True
    ):
        axis.set_title(title, fontsize=10, fontweight="bold")
    figure.suptitle(
        f"SPATCH high-entropy model, fold 0 | threshold={args.threshold:.3f}",
        y=0.992,
        fontsize=12,
        fontweight="bold",
    )
    figure.subplots_adjust(
        left=0.12,
        right=0.995,
        top=0.94,
        bottom=0.035,
        wspace=0.04,
        hspace=0.18,
    )
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=220)
    plt.close(figure)
    print(args.metrics)


if __name__ == "__main__":
    main()
