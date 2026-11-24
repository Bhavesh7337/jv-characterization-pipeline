from __future__ import annotations

import math

import numpy as np

from .config import PipelineConfig
from .types import AnalysisResult, Curve


def assign_hysteresis(curves: list[Curve], results: list[AnalysisResult]) -> None:
    by_device: dict[str, list[int]] = {}
    for index, result in enumerate(results):
        by_device.setdefault(result.device_id, []).append(index)

    for indices in by_device.values():
        forward = next(
            (
                index
                for index in indices
                if results[index].scan == "forward" and results[index].metrics is not None
            ),
            None,
        )
        reverse = next(
            (
                index
                for index in indices
                if results[index].scan == "reverse" and results[index].metrics is not None
            ),
            None,
        )
        if forward is None or reverse is None:
            continue

        forward_result = results[forward]
        reverse_result = results[reverse]
        forward_metrics = forward_result.metrics
        reverse_metrics = reverse_result.metrics
        assert forward_metrics is not None and reverse_metrics is not None

        pce_denominator = max(
            abs(forward_metrics.pce_percent), abs(reverse_metrics.pce_percent), 1e-12
        )
        pce_difference = (
            abs(forward_metrics.pce_percent - reverse_metrics.pce_percent) / pce_denominator
        )

        forward_curve = curves[forward]
        reverse_curve = curves[reverse]
        vmax = min(
            forward_metrics.voc_v,
            reverse_metrics.voc_v,
            float(forward_curve.voltage_v.max()),
            float(reverse_curve.voltage_v.max()),
        )
        grid = np.linspace(0.0, vmax, 150)
        jf = np.interp(
            grid, forward_curve.voltage_v, forward_curve.current_density_ma_cm2
        )
        jr = np.interp(
            grid, reverse_curve.voltage_v, reverse_curve.current_density_ma_cm2
        )
        area_difference = float(
            np.trapezoid(np.abs(jf - jr), grid)
            / max(np.trapezoid((np.abs(jf) + np.abs(jr)) / 2.0, grid), 1e-12)
        )
        index = max(pce_difference, area_difference)
        forward_metrics.hysteresis_index = index
        reverse_metrics.hysteresis_index = index


def classify(result: AnalysisResult, config: PipelineConfig) -> None:
    if result.metrics is None:
        result.status = "ERROR"
        return

    metrics = result.metrics
    thresholds = config.thresholds
    metric_checks = [
        (metrics.voc_v < thresholds.voc_min_v, f"Voc < {thresholds.voc_min_v:g} V"),
        (
            metrics.jsc_ma_cm2 < thresholds.jsc_min_ma_cm2,
            f"Jsc < {thresholds.jsc_min_ma_cm2:g} mA/cm²",
        ),
        (metrics.ff < thresholds.ff_min, f"FF < {thresholds.ff_min:g}"),
        (
            metrics.pce_percent < thresholds.pce_min_percent,
            f"PCE < {thresholds.pce_min_percent:g}%",
        ),
        (
            metrics.rs_ohm_cm2 > thresholds.rs_max_ohm_cm2,
            f"Rs > {thresholds.rs_max_ohm_cm2:g} Ω·cm²",
        ),
        (
            metrics.rsh_ohm_cm2 < thresholds.rsh_min_ohm_cm2,
            f"Rsh < {thresholds.rsh_min_ohm_cm2:g} Ω·cm²",
        ),
    ]
    result.failed_rules.extend(message for failed, message in metric_checks if failed)

    if metrics.ff < thresholds.ff_min:
        result.anomalies.append("poor fill factor")
    if metrics.rsh_ohm_cm2 < thresholds.rsh_min_ohm_cm2:
        result.anomalies.append("shunting")
    if (
        math.isfinite(metrics.s_kink_score)
        and metrics.s_kink_score > thresholds.s_kink_score_max
    ):
        result.anomalies.append("S-kink")
    if (
        math.isfinite(metrics.hysteresis_index)
        and metrics.hysteresis_index > thresholds.hysteresis_index_max
    ):
        result.anomalies.append("hysteresis")

    if (
        result.rf_prediction
        and math.isfinite(result.rf_anomaly_probability)
        and result.rf_anomaly_probability >= thresholds.rf_anomaly_probability_min
    ):
        result.anomalies.append(f"random forest: {result.rf_prediction}")
        if config.classification.fail_on_rf_anomaly:
            result.failed_rules.append(
                "RF anomaly probability "
                f"{result.rf_anomaly_probability:.3f} >= "
                f"{thresholds.rf_anomaly_probability_min:.3f} "
                f"({result.rf_prediction})"
            )

    fail_on = set(config.classification.fail_on_anomalies)
    for anomaly in result.anomalies:
        if anomaly in fail_on:
            result.failed_rules.append(f"Anomaly: {anomaly}")

    result.anomalies = list(dict.fromkeys(result.anomalies))
    result.failed_rules = list(dict.fromkeys(result.failed_rules))
    result.status = "FAIL" if result.failed_rules else "PASS"
