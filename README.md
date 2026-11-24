# J-V Characterization

Command-line tools for extracting photovoltaic device metrics from raw J-V measurements and screening common curve anomalies.

The package calculates Voc, Jsc, FF, PCE, Vmp, Jmp, Pmax, Rs, and Rsh. It also checks for hysteresis, S-kink behavior, shunting, and poor fill factor. Reports are written as CSV, JSON, and self-contained HTML.

## Installation

Create a virtual environment and install the package:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e .
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -e .
```

Install the optional Random Forest support with:

```bash
python -m pip install -e ".[ml]"
```

## Quick start

Analyze a file or a directory:

```bash
jv-characterize analyze path/to/jv-data --output report
```

The output directory contains:

- `characterization_results.csv`
- `characterization_results.json`
- `pass_fail_report.html`

Generate and analyze the included synthetic example:

```bash
python examples/generate_demo_data.py
jv-characterize analyze examples/demo_data --output report --config examples/config.json
```

## Input files

The reader accepts CSV, TSV, TXT, and DAT files. A file must contain voltage and either current density or current.

Common column names are detected automatically, including unit suffixes such as `Voltage (V)` and `Current Density (mA/cm2)`. Exact column names can be supplied in a configuration file when an instrument uses different headers.

For raw current measurements, set the active area:

```json
{
  "input": {
    "voltage_column": "Voltage (V)",
    "current_column": "Current (A)",
    "current_unit": "A",
    "area_cm2": 0.1
  }
}
```

The internal current convention is positive generated current at short circuit. The applied sign factor is included in every result row.

## Forward and reverse scans

Separate scans can be paired through filenames:

```text
pixel_01_forward.csv
pixel_01_reverse.csv
```

Files can also contain configured device and scan columns. Hysteresis remains empty when no valid pair is available.

## Configuration

Write the complete default configuration:

```bash
jv-characterize write-config config.json
```

Only values that need to change must appear in a custom configuration. See [`examples/config.json`](examples/config.json) for a short example.

Default pass/fail limits are intended as placeholders. Calibrate them against reviewed reference devices before production use.

## Random Forest detector

The supervised detector uses these extracted features:

- `voc_v`
- `jsc_ma_cm2`
- `ff`
- `pce_percent`
- `rs_ohm_cm2`
- `rsh_ohm_cm2`
- `s_kink_score`
- `hysteresis_index`

Prepare a historical results CSV with an `anomaly_label` column. Labels listed as normal in the configuration are treated as normal devices; every other label is treated as an anomaly class.

Train and save a model:

```bash
jv-characterize train labeled_results.csv --model detector.joblib
```

Apply it during analysis:

```bash
jv-characterize analyze new_data --output report --model detector.joblib
```

The training command prints an out-of-bag score. Use an independent test set for qualification. Only load model files from trusted sources because model deserialization is not safe for untrusted files.

## Tests

```bash
python -m unittest discover -s tests -v
```

The test suite covers metric extraction, sign normalization, filename pairing, anomaly classification, and report generation.

## Method notes

See [`docs/methodology.md`](docs/methodology.md) for formulas, assumptions, and known limitations.

