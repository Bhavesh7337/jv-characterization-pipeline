from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .types import AnalysisResult

FEATURES = [
    "voc_v",
    "jsc_ma_cm2",
    "ff",
    "pce_percent",
    "rs_ohm_cm2",
    "rsh_ohm_cm2",
    "s_kink_score",
    "hysteresis_index",
]
MODEL_FORMAT_VERSION = 1


@dataclass(slots=True)
class TrainingSummary:
    rows: int
    classes: list[str]
    oob_score: float


def _dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Random Forest support requires the optional dependency: "
            "python -m pip install 'jv-characterization[ml]'"
        ) from exc
    return joblib, RandomForestClassifier, SimpleImputer, Pipeline


def train_model(
    training_csv: Path,
    output_path: Path,
    config: PipelineConfig,
    label_column: str = "anomaly_label",
    n_estimators: int = 400,
    random_state: int = 42,
) -> TrainingSummary:
    joblib, RandomForestClassifier, SimpleImputer, Pipeline = _dependencies()
    training = pd.read_csv(training_csv)
    missing = [feature for feature in FEATURES if feature not in training.columns]
    if missing:
        raise ValueError(f"training CSV is missing features: {', '.join(missing)}")
    if label_column not in training.columns:
        raise ValueError(f"training CSV is missing label column {label_column!r}")

    usable = training.dropna(subset=[label_column]).copy()
    if len(usable) < 20:
        raise ValueError("at least 20 labeled rows are required")
    labels = usable[label_column].astype(str)
    if labels.nunique() < 2:
        raise ValueError("at least two label classes are required")

    normal_labels = {label.strip().lower() for label in config.classification.normal_labels}
    normal = labels.str.strip().str.lower().isin(normal_labels)
    if not normal.any() or normal.all():
        raise ValueError("training data must contain normal and anomalous examples")

    features = usable[FEATURES].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "forest",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_jobs=-1,
                    oob_score=True,
                ),
            ),
        ]
    )
    model.fit(features, labels)
    forest = model.named_steps["forest"]
    bundle = {
        "format_version": MODEL_FORMAT_VERSION,
        "features": FEATURES,
        "normal_labels": sorted(normal_labels),
        "model": model,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    return TrainingSummary(
        rows=len(usable),
        classes=[str(value) for value in forest.classes_],
        oob_score=float(forest.oob_score_),
    )


def load_model(path: Path) -> dict[str, Any]:
    joblib, _, _, _ = _dependencies()
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("model file is not a valid detector bundle")
    if bundle.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("model format version is not supported")
    if bundle.get("features") != FEATURES:
        raise ValueError("model feature schema does not match this package")
    return bundle


def apply_model(bundle: dict[str, Any], results: list[AnalysisResult], config: PipelineConfig) -> None:
    valid = [result for result in results if result.metrics is not None]
    if not valid:
        return
    frame = pd.DataFrame(
        [
            {feature: getattr(result.metrics, feature) for feature in FEATURES}
            for result in valid
            if result.metrics is not None
        ]
    ).replace([np.inf, -np.inf], np.nan)

    model = bundle["model"]
    probabilities = model.predict_proba(frame)
    predictions = model.predict(frame)
    classes = [str(value) for value in model.named_steps["forest"].classes_]
    normal_labels = {
        str(value).strip().lower()
        for value in bundle.get("normal_labels", config.classification.normal_labels)
    }
    normal_indices = [
        index for index, label in enumerate(classes) if label.strip().lower() in normal_labels
    ]
    anomaly_indices = [index for index in range(len(classes)) if index not in normal_indices]
    threshold = config.thresholds.rf_anomaly_probability_min

    for row, result in enumerate(valid):
        normal_probability = float(sum(probabilities[row][index] for index in normal_indices))
        anomaly_probability = float(np.clip(1.0 - normal_probability, 0.0, 1.0))
        result.rf_anomaly_probability = anomaly_probability
        if anomaly_probability >= threshold and anomaly_indices:
            best = max(anomaly_indices, key=lambda index: probabilities[row][index])
            result.rf_prediction = classes[best]
        else:
            result.rf_prediction = str(predictions[row])
