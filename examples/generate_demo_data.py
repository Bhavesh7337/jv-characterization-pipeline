from pathlib import Path

import numpy as np
import pandas as pd


def base_curve(voltage: np.ndarray, jsc: float, voc: float, exponent: float) -> np.ndarray:
    return jsc * (1 - np.sign(voltage / voc) * np.abs(voltage / voc) ** exponent)


def main() -> None:
    output = Path(__file__).with_name("demo_data")
    output.mkdir(exist_ok=True)
    voltage = np.linspace(-0.1, 1.15, 181)
    rng = np.random.default_rng(42)
    curves = {
        "reference_forward": base_curve(voltage, 24.5, 1.04, 8.0),
        "reference_reverse": base_curve(voltage, 25.0, 1.06, 9.0),
        "shunted": base_curve(voltage, 22.0, 1.00, 7.0) - 1000 * voltage / 115,
        "poor_ff": base_curve(voltage, 23.0, 0.96, 2.5),
    }
    curves["s_kink"] = base_curve(voltage, 23.5, 1.02, 7.0) - 5.8 * np.exp(
        -0.5 * ((voltage - 0.67) / 0.075) ** 2
    )

    for name, generated_current in curves.items():
        measured_current = -generated_current + rng.normal(0, 0.025, len(voltage))
        pd.DataFrame(
            {
                "Voltage (V)": voltage,
                "Current Density (mA/cm2)": measured_current,
            }
        ).to_csv(output / f"{name}.csv", index=False)
    print(output.resolve())


if __name__ == "__main__":
    main()

