from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class InputConfig:
    voltage_column: str | None = None
    current_density_column: str | None = None
    current_column: str | None = None
    area_cm2: float | None = None
    current_unit: str = "A"
    current_density_unit: str = "mA/cm2"
    scan_column: str | None = None
    device_column: str | None = None
    recursive: bool = False
    glob: str = "*.csv,*.txt,*.tsv,*.dat"


@dataclass(slots=True)
class MeasurementConfig:
    incident_power_mw_cm2: float = 100.0


@dataclass(slots=True)
class ExtractionConfig:
    fit_fraction_rsh: float = 0.10
    fit_fraction_rs: float = 0.10
    minimum_fit_points: int = 5
    minimum_points: int = 12


@dataclass(slots=True)
class ThresholdConfig:
    voc_min_v: float = 0.50
    jsc_min_ma_cm2: float = 10.0
    ff_min: float = 0.60
    pce_min_percent: float = 8.0
    rs_max_ohm_cm2: float = 20.0
    rsh_min_ohm_cm2: float = 300.0
    hysteresis_index_max: float = 0.10
    s_kink_score_max: float = 0.18
    rf_anomaly_probability_min: float = 0.60


@dataclass(slots=True)
class ClassificationConfig:
    fail_on_anomalies: list[str] = field(
        default_factory=lambda: ["hysteresis", "S-kink", "shunting", "poor fill factor"]
    )
    normal_labels: list[str] = field(
        default_factory=lambda: ["normal", "pass", "good", "0", "0.0", "false"]
    )
    fail_on_rf_anomaly: bool = True
    filename_scan_tokens: dict[str, list[str]] = field(
        default_factory=lambda: {
            "forward": ["forward", "fwd", "light-to-dark", "ltd"],
            "reverse": ["reverse", "rev", "backward", "dark-to-light", "dtl"],
        }
    )


@dataclass(slots=True)
class PipelineConfig:
    input: InputConfig = field(default_factory=InputConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None = None) -> PipelineConfig:
    data = PipelineConfig().to_dict()
    if path is not None:
        override = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(override, dict):
            raise ValueError("configuration root must be a JSON object")
        data = _merge(data, override)

    try:
        config = PipelineConfig(
            input=InputConfig(**data["input"]),
            measurement=MeasurementConfig(**data["measurement"]),
            extraction=ExtractionConfig(**data["extraction"]),
            thresholds=ThresholdConfig(**data["thresholds"]),
            classification=ClassificationConfig(**data["classification"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid configuration structure: {exc}") from exc
    validate_config(config)
    return config


def validate_config(config: PipelineConfig) -> None:
    if config.input.area_cm2 is not None and config.input.area_cm2 <= 0:
        raise ValueError("input.area_cm2 must be positive")
    if config.measurement.incident_power_mw_cm2 <= 0:
        raise ValueError("measurement.incident_power_mw_cm2 must be positive")
    if config.extraction.minimum_points < 3:
        raise ValueError("extraction.minimum_points must be at least 3")
    for name in ("fit_fraction_rsh", "fit_fraction_rs"):
        value = getattr(config.extraction, name)
        if not 0 < value <= 0.5:
            raise ValueError(f"extraction.{name} must be in (0, 0.5]")
    probability = config.thresholds.rf_anomaly_probability_min
    if not 0 <= probability <= 1:
        raise ValueError("thresholds.rf_anomaly_probability_min must be between 0 and 1")


def write_default_config(path: Path) -> None:
    path.write_text(json.dumps(PipelineConfig().to_dict(), indent=2), encoding="utf-8")
