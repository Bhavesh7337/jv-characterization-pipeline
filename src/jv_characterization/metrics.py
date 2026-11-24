from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import PipelineConfig
from .types import Curve, Metrics


def interpolate_crossing(x: np.ndarray, y: np.ndarray, target: float = 0.0) -> float:
    exact = np.flatnonzero(np.isclose(y, target, atol=1e-12))
    if len(exact):
        candidates = x[exact]
        positive = candidates[candidates >= 0]
        return float(positive[0] if len(positive) else candidates[0])

    crossings = np.flatnonzero((y[:-1] - target) * (y[1:] - target) < 0)
    candidates: list[float] = []
    for index in crossings:
        value = x[index] + (target - y[index]) * (x[index + 1] - x[index]) / (
            y[index + 1] - y[index]
        )
        if value >= 0:
            candidates.append(float(value))
    return min(candidates) if candidates else math.nan


def local_slope(
    voltage: np.ndarray,
    current_density: np.ndarray,
    center: float,
    span: float,
    minimum_points: int,
) -> float:
    indices = np.flatnonzero(np.abs(voltage - center) <= span)
    if len(indices) < minimum_points:
        indices = np.argsort(np.abs(voltage - center))[: min(minimum_points, len(voltage))]
    if len(indices) < 2 or np.ptp(voltage[indices]) == 0:
        return math.nan
    return float(np.polyfit(voltage[indices], current_density[indices], 1)[0])


def smooth(values: np.ndarray) -> np.ndarray:
    if len(values) < 7:
        return pd.Series(values).rolling(5, center=True, min_periods=1).mean().to_numpy()
    window = min(31, len(values) if len(values) % 2 else len(values) - 1)
    return savgol_filter(values, window_length=max(window, 5), polyorder=3)


def calculate_s_kink_score(
    voltage: np.ndarray, current_density: np.ndarray, voc: float, jsc: float
) -> float:
    mask = (voltage >= 0.15 * voc) & (voltage <= 0.95 * voc)
    if mask.sum() < 8 or voc <= 0 or jsc <= 0:
        return math.nan

    grid = np.linspace(0.15 * voc, 0.95 * voc, 120)
    normalized_voltage = grid / voc
    normalized_current = np.interp(grid, voltage, current_density) / jsc
    smoothed = smooth(normalized_current)
    slope = np.gradient(smoothed, normalized_voltage)
    curvature = np.gradient(slope, normalized_voltage)

    positive_slope = float(np.mean(np.clip(slope - 0.05, 0, None)))
    positive_curvature = float(np.mean(np.clip(curvature - 0.50, 0, None)))
    return float(min(1.0, 2.0 * positive_slope + positive_curvature / 10.0))


def extract_metrics(curve: Curve, config: PipelineConfig) -> Metrics:
    voltage = curve.voltage_v
    current_density = curve.current_density_ma_cm2

    if voltage.min() <= 0 <= voltage.max():
        jsc = float(np.interp(0.0, voltage, current_density))
    else:
        nearest = np.argsort(np.abs(voltage))[: min(5, len(voltage))]
        jsc = float(np.polyval(np.polyfit(voltage[nearest], current_density[nearest], 1), 0))
        curve.warnings.append("voltage sweep does not bracket 0 V; Jsc was extrapolated")

    voc = interpolate_crossing(voltage, current_density)
    if not np.isfinite(voc) or voc <= 0:
        raise ValueError("no positive-voltage zero-current crossing found")
    if jsc <= 0:
        raise ValueError("Jsc is not positive after sign normalization")

    generating = (voltage >= 0) & (voltage <= voc) & (current_density >= 0)
    if generating.sum() < 3:
        raise ValueError("insufficient points in the power-generating quadrant")
    qv = voltage[generating]
    qj = current_density[generating]
    power = qv * qj
    maximum = int(np.argmax(power))
    vmp = float(qv[maximum])
    jmp = float(qj[maximum])
    pmax = float(power[maximum])

    ff = pmax / (voc * jsc)
    pce = 100.0 * pmax / config.measurement.incident_power_mw_cm2
    fit_span = max(voc, float(np.ptp(voltage)))
    slope_zero = local_slope(
        voltage,
        current_density,
        0.0,
        fit_span * config.extraction.fit_fraction_rsh,
        config.extraction.minimum_fit_points,
    )
    slope_voc = local_slope(
        voltage,
        current_density,
        voc,
        fit_span * config.extraction.fit_fraction_rs,
        config.extraction.minimum_fit_points,
    )
    rsh = -1000.0 / slope_zero if slope_zero < 0 else math.inf
    rs = -1000.0 / slope_voc if slope_voc < 0 else math.inf

    return Metrics(
        voc_v=voc,
        jsc_ma_cm2=jsc,
        ff=ff,
        pce_percent=pce,
        pmax_mw_cm2=pmax,
        vmp_v=vmp,
        jmp_ma_cm2=jmp,
        rs_ohm_cm2=rs,
        rsh_ohm_cm2=rsh,
        s_kink_score=calculate_s_kink_score(voltage, current_density, voc, jsc),
    )
