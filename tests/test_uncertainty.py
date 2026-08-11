from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rank_uncertain_patches import (
    normalized_binary_entropy,
    rank_patches,
    select_unique_sources,
    top_fraction_mean,
)


class UncertaintySelectionTest(unittest.TestCase):
    def test_normalized_entropy_extrema(self) -> None:
        entropy = normalized_binary_entropy(np.array([0.0, 0.5, 1.0]))
        self.assertAlmostEqual(float(entropy[0]), 0.0, places=12)
        self.assertAlmostEqual(float(entropy[1]), 1.0, places=12)
        self.assertAlmostEqual(float(entropy[2]), 0.0, places=12)

    def test_top_fraction_score(self) -> None:
        values = np.arange(10, dtype=float)
        self.assertEqual(top_fraction_mean(values, 0.2), 8.5)

    def test_ranking_and_source_diversity(self) -> None:
        probabilities = np.array(
            [
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.49, 0.51], [0.49, 0.51]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )
        ranked = rank_patches(
            probabilities,
            ["p1", "p2", "p3"],
            ["source_a", "source_a", "source_b"],
            top_fraction=1.0,
        )
        selected = select_unique_sources(ranked, budget=2)
        self.assertEqual([row.patch_id for row in selected], ["p1", "p3"])


if __name__ == "__main__":
    unittest.main()
