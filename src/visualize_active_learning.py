"""Plot the equal-budget active-selection comparison."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {"Baseline": "#707784", "High entropy": "#087CB7", "Random": "#E26600"}
DATASETS = ("BBBC039", "SPATCH DAPI")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("finetune", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selection = {row["dataset"]: row for row in read_rows(args.selection)}
    finetune = read_rows(args.finetune)

    display_key = {"BBBC039": "BBBC039", "SPATCH DAPI": "SPATCH_DAPI"}
    chosen: dict[tuple[str, str], dict[str, str]] = {}
    for display, key in display_key.items():
        chosen[(display, "Baseline")] = next(
            row for row in finetune if row["dataset"] == key and row["method"] == "baseline"
        )
        ratio = "4_to_20" if key == "BBBC039" else "20_to_20"
        chosen[(display, "High entropy")] = next(
            row
            for row in finetune
            if row["dataset"] == key
            and row["method"] == "high_entropy_20"
            and row["target_external_sampling"] == ratio
        )
        chosen[(display, "Random")] = next(
            row
            for row in finetune
            if row["dataset"] == key
            and row["method"] == "random_20"
            and row["target_external_sampling"] == ratio
        )

    figure, axes = plt.subplots(1, 3, figsize=(12.8, 4.2))
    x = np.arange(len(DATASETS))
    width = 0.32

    high_error = [100 * float(selection[display_key[name]]["high_entropy_mean_pixel_error"]) for name in DATASETS]
    random_error = [100 * float(selection[display_key[name]]["random_mean_pixel_error"]) for name in DATASETS]
    axes[0].bar(x - width / 2, high_error, width, label="High entropy", color=COLORS["High entropy"])
    axes[0].bar(x + width / 2, random_error, width, label="Random", color=COLORS["Random"])
    axes[0].set_title("Selected-patch pixel error")
    axes[0].set_ylabel("Mean pixel error (%)")
    axes[0].set_xticks(x, DATASETS)
    for index, name in enumerate(DATASETS):
        rho = float(selection[display_key[name]]["entropy_error_spearman_rho"])
        axes[0].text(index, max(high_error[index], random_error[index]) + 1.1, f"rho={rho:.3f}", ha="center", fontsize=9)

    methods = ("Baseline", "High entropy", "Random")
    offsets = (-width, 0, width)
    for method, offset in zip(methods, offsets):
        dice = [float(chosen[(dataset, method)]["Dice"]) for dataset in DATASETS]
        axes[1].bar(x + offset, dice, width, label=method, color=COLORS[method])
    axes[1].set_title("Fold-0 held-out Dice")
    axes[1].set_ylim(0.895, 0.914)
    axes[1].set_xticks(x, DATASETS)

    for method, offset in zip(methods, offsets):
        worst_error = [
            100
            * max(
                float(chosen[(dataset, method)]["FNR"]),
                float(chosen[(dataset, method)]["FDR"]),
            )
            for dataset in DATASETS
        ]
        axes[2].bar(x + offset, worst_error, width, label=method, color=COLORS[method])
    axes[2].axhline(10, color="#23313D", linewidth=1, linestyle="--")
    axes[2].set_title("Fold-0 worst pixel error")
    axes[2].set_ylabel("max(FNR, FDR) (%)")
    axes[2].set_xticks(x, DATASETS)

    for axis in axes:
        axis.grid(axis="y", color="#D8DDE2", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    figure.suptitle("Normalized predictive entropy: equal-budget sample selection", y=1.04, fontsize=13, fontweight="bold")
    figure.text(0.5, 0.01, "20 external patches per strategy; SPATCH uses 1:1 target/external sampling", ha="center", fontsize=9, color="#4B5560")
    figure.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.19, wspace=0.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
