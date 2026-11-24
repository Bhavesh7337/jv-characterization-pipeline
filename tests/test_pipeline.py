import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from jv_characterization.pipeline import analyze


class PipelineTests(unittest.TestCase):
    def test_end_to_end_reports_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            report = root / "report"
            data.mkdir()
            voltage = np.linspace(-0.1, 1.15, 181)
            current = -(24.0 * (1 - np.sign(voltage / 1.02) * np.abs(voltage / 1.02) ** 8))
            pd.DataFrame(
                {"Voltage (V)": voltage, "Current Density (mA/cm2)": current}
            ).to_csv(data / "reference.csv", index=False)

            results = analyze(data, report)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "PASS")
            self.assertTrue((report / "characterization_results.csv").is_file())
            self.assertTrue((report / "pass_fail_report.html").is_file())
            parsed = json.loads((report / "characterization_results.json").read_text())
            self.assertEqual(parsed[0]["status"], "PASS")


if __name__ == "__main__":
    unittest.main()

