from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class Curve:
    source: Path
    device_id: str
    scan: str
    voltage_v: np.ndarray
    current_density_ma_cm2: np.ndarray
    sign_factor: int
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Metrics:
    voc_v: float
    jsc_ma_cm2: float
    ff: float
    pce_percent: float
    pmax_mw_cm2: float
    vmp_v: float
    jmp_ma_cm2: float
    rs_ohm_cm2: float
    rsh_ohm_cm2: float
    s_kink_score: float
    hysteresis_index: float = math.nan


@dataclass(slots=True)
class AnalysisResult:
    source: Path
    device_id: str
    scan: str
    points: int
    sign_factor: int
    metrics: Metrics | None = None
    rf_prediction: str = ""
    rf_anomaly_probability: float = math.nan
    anomalies: list[str] = field(default_factory=list)
    failed_rules: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    status: str = "ERROR"
    error: str = ""

