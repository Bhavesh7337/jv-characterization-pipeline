import math
import unittest
from pathlib import Path

import numpy as np

from jv_characterization.config import PipelineConfig
from jv_characterization.metrics import calculate_s_kink_score, extract_metrics
from jv_characterization.types import Curve


def curve_from(voltage: np.ndarray, current: np.ndarray) -> Curve:
    return Curve(Path("test.csv"), "test", "unknown", voltage, current, 1)


class MetricsTests(unittest.TestCase):
    def test_reference_curve(self) -> None:
        voltage = np.linspace(-0.1, 1.15, 251)
        current = 24.5 * (1 - np.sign(voltage / 1.04) * np.abs(voltage / 1.04) ** 8)
        metrics = extract_metrics(curve_from(voltage, current), PipelineConfig())

        self.assertAlmostEqual(metrics.voc_v, 1.04, places=3)
        self.assertAlmostEqual(metrics.jsc_ma_cm2, 24.5, places=2)
        self.assertGreater(metrics.ff, 0.65)
        self.assertLess(metrics.ff, 0.72)
        self.assertGreater(metrics.rsh_ohm_cm2, 1000)
        self.assertLess(metrics.rs_ohm_cm2, 10)

    def test_s_kink_score_separates_local_reversal(self) -> None:
        voltage = np.linspace(-0.1, 1.15, 251)
        normal = 24.0 * (1 - np.sign(voltage / 1.03) * np.abs(voltage / 1.03) ** 8)
        kinked = normal - 6.0 * np.exp(-0.5 * ((voltage - 0.67) / 0.075) ** 2)

        normal_score = calculate_s_kink_score(voltage, normal, 1.03, 24.0)
        kink_score = calculate_s_kink_score(voltage, kinked, 1.03, 24.0)
        self.assertLess(normal_score, 0.05)
        self.assertGreater(kink_score, 0.18)
        self.assertTrue(math.isfinite(kink_score))


if __name__ == "__main__":
    unittest.main()

