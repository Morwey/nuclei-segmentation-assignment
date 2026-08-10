"""Recompute ROI metrics and figures from the saved binary masks."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from common import binary_metrics, load_rois, macro_average, pooled_metrics


METHODS = {
    "cellpose_zero_shot": {
        "label": "CPSAM-v2\nzero-shot",
        "color": "#6B7280",
    },
    "cellpose_finetuned_loo": {
        "label": "CPSAM-v2\nfine-tuned",
        "color": "#0072B2",
    },
    "nnunet_loo": {
        "label": "nnU-Net v2\nfrom scratch",
        "color": "#D55E00",
    },
}


def prediction_name(roi_name: str) -> str:
    return f"{roi_name}-dapi_mask.tif"


def load_results(
    annotation_dir: Path, prediction_root: Path
) -> tuple[list[dict[str, object]], dict[str, list[np.ndarray]]]:
    rois = load_rois(annotation_dir)
    rows: list[dict[str, object]] = []
    masks: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}

    for method in METHODS:
        for roi in rois:
            path = prediction_root / method / prediction_name(roi.name)
            if not path.exists():
                raise FileNotFoundError(f"Missing prediction: {path}")
            prediction = np.asarray(tifffile.imread(path)) > 0
            row = binary_metrics(prediction, roi.mask)
            rows.append({"method": method, "roi": roi.name, **row})
            masks[method].append(prediction)
    return rows, masks


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        macro = macro_average(method_rows)  # type: ignore[arg-type]
        pooled = pooled_metrics(method_rows)  # type: ignore[arg-type]
        summary.append(
            {
                "method": method,
                "aggregation": "macro",
                **macro,
            }
        )
        summary.append(
            {
                "method": method,
                "aggregation": "pooled",
                **pooled,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_metric_figure(summary: list[dict[str, object]], path: Path) -> None:
    macro = {row["method"]: row for row in summary if row["aggregation"] == "macro"}
    methods = list(METHODS)
    x = np.arange(len(methods))
    colors = [METHODS[method]["color"] for method in methods]
    labels = [METHODS[method]["label"] for method in methods]

    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.45))
    panels = (("Dice", "Dice", (0.88, 0.94)), ("FNR", "FNR (%)", (0, 14)), ("FDR", "FDR (%)", (0, 14)))
    for ax, (key, title, ylim) in zip(axes, panels):
        values = [float(macro[method][key]) for method in methods]
        shown = values if key == "Dice" else [100 * value for value in values]
        bars = ax.bar(x, shown, color=colors, width=0.68)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x, labels)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        if key != "Dice":
            ax.axhline(10, color="#111827", linestyle="--", linewidth=1.0, label="10% target")
        for bar, value in zip(bars, shown):
            text = f"{value:.3f}" if key == "Dice" else f"{value:.1f}"
            ax.text(bar.get_x() + bar.get_width() / 2, value + (0.001 if key == "Dice" else 0.25), text, ha="center", va="bottom", fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    figure.suptitle("Five-fold leave-one-ROI-out evaluation (macro average)", fontsize=11, fontweight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def normalized(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    return np.clip((image.astype(np.float32) - low) / max(float(high - low), 1.0), 0, 1)


def draw_overlay(ax: plt.Axes, image: np.ndarray, mask: np.ndarray) -> None:
    ax.imshow(normalized(image), cmap="gray", vmin=0, vmax=1)
    ax.contour(mask.astype(np.uint8), levels=[0.5], colors=["#00D8A4"], linewidths=0.65)
    ax.set_xticks([])
    ax.set_yticks([])


def make_roi_figure(
    annotation_dir: Path,
    rows: list[dict[str, object]],
    masks: dict[str, list[np.ndarray]],
    path: Path,
) -> None:
    rois = load_rois(annotation_dir)
    methods = list(METHODS)
    figure, axes = plt.subplots(len(rois), 2 + len(methods), figsize=(12.0, 12.7))
    column_titles = ["DAPI", "Manual mask"] + [METHODS[method]["label"].replace("\n", " ") for method in methods]

    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=9, fontweight="bold")

    for index, roi in enumerate(rois):
        axes[index, 0].imshow(normalized(roi.image), cmap="gray", vmin=0, vmax=1)
        axes[index, 0].set_ylabel(f"ROI {index + 1}", fontsize=9, fontweight="bold")
        axes[index, 0].set_xticks([])
        axes[index, 0].set_yticks([])
        axes[index, 1].imshow(roi.mask, cmap="gray", vmin=0, vmax=1)
        axes[index, 1].set_xticks([])
        axes[index, 1].set_yticks([])

        for offset, method in enumerate(methods, start=2):
            prediction = masks[method][index]
            draw_overlay(axes[index, offset], roi.image, prediction)
            row = next(item for item in rows if item["method"] == method and item["roi"] == roi.name)
            axes[index, offset].set_xlabel(
                f"D {float(row['Dice']):.3f} | FN {100 * float(row['FNR']):.1f}% | FD {100 * float(row['FDR']):.1f}%",
                fontsize=7,
            )

    figure.suptitle("Held-out ROI predictions; green line = predicted boundary", y=0.994, fontsize=11, fontweight="bold")
    figure.subplots_adjust(left=0.035, right=0.995, top=0.945, bottom=0.015, wspace=0.06, hspace=0.24)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=240)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations"))
    parser.add_argument("--predictions", type=Path, default=Path("results/roi_predictions"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, masks = load_results(args.annotations, args.predictions)
    summary = summarize(rows)
    write_csv(args.output_dir / "metrics_per_roi.csv", rows)
    write_csv(args.output_dir / "metrics_summary.csv", summary)
    (args.output_dir / "metrics.json").write_text(
        json.dumps({"per_roi": rows, "summary": summary}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    make_metric_figure(summary, args.output_dir / "figures" / "metric_comparison.png")
    make_roi_figure(
        args.annotations,
        rows,
        masks,
        args.output_dir / "figures" / "roi_comparison.png",
    )
    print(args.output_dir / "metrics_summary.csv")


if __name__ == "__main__":
    main()
