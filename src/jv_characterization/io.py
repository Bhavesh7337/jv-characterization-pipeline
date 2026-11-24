from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .types import Curve

VOLTAGE_ALIASES = ("voltage", "volt", "bias", "potential", "v")
CURRENT_DENSITY_ALIASES = (
    "currentdensity",
    "current_density",
    "jma/cm2",
    "j(ma/cm2)",
    "ma/cm2",
    "current density",
    "j",
)
CURRENT_ALIASES = ("current", "current(a)", "i(a)", "amps", "ampere", "i")


def clean_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def find_column(
    columns: Iterable[object], requested: str | None, aliases: Iterable[str]
) -> object | None:
    columns = list(columns)
    if requested:
        requested_name = clean_name(requested)
        for column in columns:
            if str(column) == requested or clean_name(column) == requested_name:
                return column
        raise ValueError(f"configured column {requested!r} was not found")

    lookup = {clean_name(column): column for column in columns}
    normalized_aliases = sorted((clean_name(alias) for alias in aliases), key=len, reverse=True)
    for alias in normalized_aliases:
        if alias in lookup:
            return lookup[alias]
    for alias in normalized_aliases:
        if len(alias) < 2:
            continue
        for normalized, original in lookup.items():
            if normalized.startswith(alias):
                return original
    return None


def _detect_header_and_delimiter(path: Path) -> tuple[int, str | None]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    candidates = [
        line
        for line in lines[:80]
        if line.strip() and not line.lstrip().startswith(("#", ";"))
    ]
    for index, line in enumerate(lines[:80]):
        if not line.strip() or line.lstrip().startswith(("#", ";")):
            continue
        normalized = clean_name(line)
        if any(token in normalized for token in ("voltage", "volt", "bias", "current", "potential")):
            try:
                delimiter = csv.Sniffer().sniff(
                    "\n".join(candidates[:10]), delimiters=",;\t "
                ).delimiter
            except csv.Error:
                delimiter = None
            return index, delimiter
    return 0, None


def read_table(path: Path) -> pd.DataFrame:
    header_row, delimiter = _detect_header_and_delimiter(path)
    separator = r"\s+" if delimiter == " " else delimiter
    try:
        frame = pd.read_csv(
            path,
            skiprows=header_row,
            comment="#",
            sep=separator,
            engine="python",
        )
    except Exception as exc:
        raise ValueError(f"could not parse table: {exc}") from exc

    if frame.shape[1] < 2:
        raise ValueError("expected at least two columns")

    try:
        float(str(frame.columns[0]))
        float(str(frame.columns[1]))
    except (TypeError, ValueError):
        return frame

    frame = pd.read_csv(path, header=None, comment="#", sep=separator, engine="python")
    frame.columns = ["Voltage", "Current"] + [
        f"Column_{index}" for index in range(3, frame.shape[1] + 1)
    ]
    return frame


def infer_scan_and_device(path: Path, config: PipelineConfig) -> tuple[str, str]:
    stem = path.stem
    normalized = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")
    scan = "unknown"
    device = stem

    for label, tokens in config.classification.filename_scan_tokens.items():
        for token in tokens:
            normalized_token = re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_")
            if re.search(rf"(?:^|_){re.escape(normalized_token)}(?:_|$)", normalized):
                scan = label
            token_pattern = re.sub(r"[^a-z0-9]+", r"[^a-z0-9]*", token.lower())
            device = re.sub(rf"(?i)[_\- ]*{token_pattern}[_\- ]*", "_", device)

    device = re.sub(r"_+", "_", device).strip("_-. ") or stem
    return scan, device


def _current_to_density(values: np.ndarray, unit: str, area_cm2: float | None) -> np.ndarray:
    if area_cm2 is None or area_cm2 <= 0:
        raise ValueError("input.area_cm2 is required when using a current column")
    scale = {"a": 1000.0, "ma": 1.0, "ua": 0.001, "µa": 0.001, "μa": 0.001}.get(
        unit.strip().lower()
    )
    if scale is None:
        raise ValueError("current_unit must be A, mA, uA, µA, or μA")
    return values * scale / area_cm2


def _density_to_ma_cm2(values: np.ndarray, unit: str) -> np.ndarray:
    normalized = clean_name(unit.replace("µ", "u").replace("μ", "u").replace("²", "2"))
    scale = {"macm2": 1.0, "acm2": 1000.0, "uacm2": 0.001}.get(normalized)
    if scale is None:
        raise ValueError("current_density_unit must be A/cm2, mA/cm2, or uA/cm2")
    return values * scale


def _normalize_scan(value: object) -> str:
    text = str(value).strip().lower()
    if text.startswith(("f", "l")):
        return "forward"
    if text.startswith(("r", "b", "d")):
        return "reverse"
    return text or "unknown"


def load_curves(path: Path, config: PipelineConfig) -> list[Curve]:
    frame = read_table(path)
    settings = config.input
    voltage_column = find_column(frame.columns, settings.voltage_column, VOLTAGE_ALIASES)
    density_column = find_column(
        frame.columns, settings.current_density_column, CURRENT_DENSITY_ALIASES
    )
    current_column = find_column(frame.columns, settings.current_column, CURRENT_ALIASES)

    if voltage_column is None:
        voltage_column = frame.columns[0]
    if density_column is None and current_column is None:
        if frame.shape[1] != 2:
            raise ValueError(
                "could not identify a current column; set input.current_column or "
                "input.current_density_column"
            )
        if settings.area_cm2 is None:
            density_column = frame.columns[1]
        else:
            current_column = frame.columns[1]

    scan_column = find_column(frame.columns, settings.scan_column, ("scan", "direction", "sweep"))
    device_column = find_column(
        frame.columns, settings.device_column, ("device", "deviceid", "sample", "pixel")
    )
    default_scan, default_device = infer_scan_and_device(path, config)
    group_columns = [column for column in (device_column, scan_column) if column is not None]
    groups = frame.groupby(group_columns, dropna=False) if group_columns else [(None, frame)]

    curves: list[Curve] = []
    for _, group in groups:
        voltage = pd.to_numeric(group[voltage_column], errors="coerce").to_numpy(float)
        if density_column is not None:
            raw = pd.to_numeric(group[density_column], errors="coerce").to_numpy(float)
            current_density = _density_to_ma_cm2(raw, settings.current_density_unit)
        else:
            raw = pd.to_numeric(group[current_column], errors="coerce").to_numpy(float)
            current_density = _current_to_density(raw, settings.current_unit, settings.area_cm2)

        finite = np.isfinite(voltage) & np.isfinite(current_density)
        voltage = voltage[finite]
        current_density = current_density[finite]
        if len(voltage) < config.extraction.minimum_points:
            raise ValueError(
                f"only {len(voltage)} numeric rows; at least "
                f"{config.extraction.minimum_points} are required"
            )

        order = np.argsort(voltage)
        collapsed = (
            pd.DataFrame({"voltage": voltage[order], "current_density": current_density[order]})
            .groupby("voltage", as_index=False)
            .mean()
        )
        voltage = collapsed["voltage"].to_numpy(float)
        current_density = collapsed["current_density"].to_numpy(float)
        near_zero = np.argsort(np.abs(voltage))[: min(5, len(voltage))]
        sign_factor = 1 if np.median(current_density[near_zero]) >= 0 else -1
        current_density = current_density * sign_factor

        device = str(group[device_column].iloc[0]) if device_column is not None else default_device
        scan = _normalize_scan(group[scan_column].iloc[0]) if scan_column is not None else default_scan
        curves.append(
            Curve(
                source=path,
                device_id=device,
                scan=scan,
                voltage_v=voltage,
                current_density_ma_cm2=current_density,
                sign_factor=sign_factor,
            )
        )
    return curves


def discover_files(input_path: Path, config: PipelineConfig) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    patterns = [part.strip() for part in config.input.glob.split(",") if part.strip()]
    finder = input_path.rglob if config.input.recursive else input_path.glob
    return sorted({path for pattern in patterns for path in finder(pattern) if path.is_file()})
