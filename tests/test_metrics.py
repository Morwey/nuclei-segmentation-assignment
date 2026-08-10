from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import binary_metrics, macro_average, pooled_metrics


class BinaryMetricsTest(unittest.TestCase):
    def test_perfect_prediction(self) -> None:
        mask = np.array([[0, 1], [1, 0]], dtype=bool)
        result = binary_metrics(mask, mask)
        self.assertEqual(result["Dice"], 1.0)
        self.assertEqual(result["FNR"], 0.0)
        self.assertEqual(result["FDR"], 0.0)

    def test_fnr_and_fdr_denominators(self) -> None:
        truth = np.array([[1, 1], [0, 0]], dtype=bool)
        prediction = np.array([[1, 0], [1, 0]], dtype=bool)
        result = binary_metrics(prediction, truth)
        self.assertEqual(result["TP"], 1)
        self.assertEqual(result["FP"], 1)
        self.assertEqual(result["FN"], 1)
        self.assertEqual(result["FNR"], 0.5)
        self.assertEqual(result["FDR"], 0.5)

    def test_macro_and_pooled_are_separate(self) -> None:
        first = binary_metrics(np.ones((1, 10), dtype=bool), np.ones((1, 10), dtype=bool))
        second = binary_metrics(np.zeros((1, 1), dtype=bool), np.ones((1, 1), dtype=bool))
        macro = macro_average([first, second])
        pooled = pooled_metrics([first, second])
        self.assertAlmostEqual(float(macro["Dice"]), 0.5)
        self.assertAlmostEqual(float(pooled["Dice"]), 20 / 21)


if __name__ == "__main__":
    unittest.main()
