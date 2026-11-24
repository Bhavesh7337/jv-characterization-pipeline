import unittest
from pathlib import Path

from jv_characterization.anomalies import classify
from jv_characterization.config import PipelineConfig
from jv_characterization.types import AnalysisResult, Metrics


class ClassificationTests(unittest.TestCase):
    def test_shunted_device_fails(self) -> None:
        metrics = Metrics(
            voc_v=0.9,
            jsc_ma_cm2=20.0,
            ff=0.48,
            pce_percent=9.0,
            pmax_mw_cm2=9.0,
            vmp_v=0.6,
            jmp_ma_cm2=15.0,
            rs_ohm_cm2=8.0,
            rsh_ohm_cm2=120.0,
            s_kink_score=0.01,
        )
        result = AnalysisResult(Path("test.csv"), "test", "unknown", 100, 1, metrics)
        classify(result, PipelineConfig())

        self.assertEqual(result.status, "FAIL")
        self.assertIn("shunting", result.anomalies)
        self.assertIn("poor fill factor", result.anomalies)

    def test_healthy_device_passes(self) -> None:
        metrics = Metrics(
            voc_v=1.0,
            jsc_ma_cm2=23.0,
            ff=0.70,
            pce_percent=16.1,
            pmax_mw_cm2=16.1,
            vmp_v=0.8,
            jmp_ma_cm2=20.1,
            rs_ohm_cm2=5.0,
            rsh_ohm_cm2=900.0,
            s_kink_score=0.02,
        )
        result = AnalysisResult(Path("test.csv"), "test", "unknown", 100, 1, metrics)
        classify(result, PipelineConfig())
        self.assertEqual(result.status, "PASS")


if __name__ == "__main__":
    unittest.main()

