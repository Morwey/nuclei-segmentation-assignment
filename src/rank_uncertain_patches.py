"""Rank unlabeled probability maps by normalized predictive entropy."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RankedPatch:
    patch_id: str
    source_id: str
    entropy_score: float


def normalized_binary_entropy(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be between 0 and 1")
    epsilon = np.finfo(np.float64).eps
    clipped = np.clip(probability, epsilon, 1 - epsilon)
    return -(
        clipped * np.log(clipped) + (1 - clipped) * np.log(1 - clipped)
    ) / np.log(2)


def top_fraction_mean(values: np.ndarray, fraction: float = 0.1) -> float:
    flattened = np.asarray(values, dtype=np.float64).ravel()
    if flattened.size == 0:
        raise ValueError("Entropy map is empty")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    count = max(1, int(np.ceil(flattened.size * fraction)))
    split = flattened.size - count
    return float(np.mean(np.partition(flattened, split)[split:]))


def rank_patches(
    probabilities: np.ndarray,
    patch_ids: list[str],
    source_ids: list[str],
    top_fraction: float = 0.1,
) -> list[RankedPatch]:
    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 3:
        raise ValueError("probabilities must have shape (N, Y, X)")
    if len(probabilities) != len(patch_ids) or len(patch_ids) != len(source_ids):
        raise ValueError("probabilities, patch_ids and source_ids must have equal length")
    rows = [
        RankedPatch(
            patch_id=str(patch_id),
            source_id=str(source_id),
            entropy_score=top_fraction_mean(
                normalized_binary_entropy(probability), top_fraction
            ),
        )
        for probability, patch_id, source_id in zip(
            probabilities, patch_ids, source_ids
        )
    ]
    return sorted(rows, key=lambda row: (-row.entropy_score, row.patch_id))


def select_unique_sources(
    ranked: list[RankedPatch], budget: int
) -> list[RankedPatch]:
    if budget < 1:
        raise ValueError("budget must be positive")
    selected: list[RankedPatch] = []
    used_sources: set[str] = set()
    for row in ranked:
        if row.source_id in used_sources:
            continue
        selected.append(row)
        used_sources.add(row.source_id)
        if len(selected) == budget:
            break
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="NPZ with probabilities (N,Y,X), patch_ids and source_ids arrays.",
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = np.load(args.input, allow_pickle=True)
    ranked = rank_patches(
        payload["probabilities"],
        [str(value) for value in payload["patch_ids"]],
        [str(value) for value in payload["source_ids"]],
        args.top_fraction,
    )
    selected = select_unique_sources(ranked, args.budget)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("rank", "patch_id", "source_id", "entropy_score")
        )
        writer.writeheader()
        for rank, row in enumerate(selected, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "patch_id": row.patch_id,
                    "source_id": row.source_id,
                    "entropy_score": f"{row.entropy_score:.8f}",
                }
            )
    print(args.output)


if __name__ == "__main__":
    main()
