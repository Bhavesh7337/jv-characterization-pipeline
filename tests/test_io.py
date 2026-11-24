import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from jv_characterization.config import PipelineConfig
from jv_characterization.io import infer_scan_and_device, load_curves


class InputTests(unittest.TestCase):
    def test_unit_header_and_negative_current_sign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pixel_01_forward.csv"
            voltage = np.linspace(-0.1, 1.1, 25)
            current = -(22.0 * (1 - voltage / 0.95))
            pd.DataFrame(
                {"Voltage (V)": voltage, "Current Density (mA/cm2)": current}
            ).to_csv(path, index=False)

            curve = load_curves(path, PipelineConfig())[0]
            self.assertEqual(curve.sign_factor, -1)
            self.assertGreater(np.interp(0, curve.voltage_v, curve.current_density_ma_cm2), 0)
            self.assertEqual(curve.device_id, "pixel_01")
            self.assertEqual(curve.scan, "forward")

    def test_scan_name_pairing(self) -> None:
        config = PipelineConfig()
        self.assertEqual(
            infer_scan_and_device(Path("sample_reverse.csv"), config),
            ("reverse", "sample"),
        )


if __name__ == "__main__":
    unittest.main()

