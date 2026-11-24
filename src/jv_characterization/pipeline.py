from __future__ import annotations

from pathlib import Path

import numpy as np

from .anomalies import assign_hysteresis, classify
from .config import PipelineConfig
from .io import discover_files, infer_scan_and_device, load_curves
from .metrics import extract_metrics
from .model import apply_model, load_model
from .reporting import write_reports
from .types import AnalysisResult, Curve


def analyze(
    input_path: Path,
    output_dir: Path,
    config: PipelineConfig | None = None,
    model_path: Path | None = None,
) -> list[AnalysisResult]:
    config = config or PipelineConfig()
    paths = discover_files(input_path, config)
    if not paths:
        raise ValueError("no matching J-V files found")

    curves: list[Curve] = []
    results: list[AnalysisResult] = []
    for path in paths:
        try:
            loaded = load_curves(path, config)
            for curve in loaded:
                curves.append(curve)
                try:
                    metrics = extract_metrics(curve, config)
                    results.append(
                        AnalysisResult(
                            source=curve.source,
                            device_id=curve.device_id,
                            scan=curve.scan,
                            points=len(curve.voltage_v),
                            sign_factor=curve.sign_factor,
                            metrics=metrics,
                            warnings=list(curve.warnings),
                        )
                    )
                except Exception as exc:
                    results.append(
                        AnalysisResult(
                            source=curve.source,
                            device_id=curve.device_id,
                            scan=curve.scan,
                            points=len(curve.voltage_v),
                            sign_factor=curve.sign_factor,
                            warnings=list(curve.warnings),
                            error=str(exc),
                        )
                    )
        except Exception as exc:
            scan, device = infer_scan_and_device(path, config)
            curves.append(
                Curve(
                    source=path,
                    device_id=device,
                    scan=scan,
                    voltage_v=np.array([], dtype=float),
                    current_density_ma_cm2=np.array([], dtype=float),
                    sign_factor=1,
                )
            )
            results.append(
                AnalysisResult(
                    source=path,
                    device_id=device,
                    scan=scan,
                    points=0,
                    sign_factor=1,
                    error=str(exc),
                )
            )

    assign_hysteresis(curves, results)
    if model_path is not None:
        apply_model(load_model(model_path), results, config)
    for result in results:
        classify(result, config)
    write_reports(output_dir, curves, results, config)
    return results
