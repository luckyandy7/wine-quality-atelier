from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "winequality-red.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "wine_quality_bundle.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]

FEATURE_LABELS = {
    "fixed acidity": "Fixed acidity",
    "volatile acidity": "Volatile acidity",
    "citric acid": "Citric acid",
    "residual sugar": "Residual sugar",
    "chlorides": "Chlorides",
    "free sulfur dioxide": "Free sulfur dioxide",
    "total sulfur dioxide": "Total sulfur dioxide",
    "density": "Density",
    "pH": "pH",
    "sulphates": "Sulphates",
    "alcohol": "Alcohol",
}


@dataclass(frozen=True)
class Prediction:
    high_quality_probability: float
    predicted_class: int
    predicted_quality: float
    tier: str
    notes: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "highQualityProbability": round(self.high_quality_probability, 4),
            "predictedClass": self.predicted_class,
            "predictedQuality": round(self.predicted_quality, 2),
            "tier": self.tier,
            "notes": self.notes,
        }


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    missing = [name for name in FEATURE_NAMES + ["quality"] if name not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    return df


def _feature_specs(df: pd.DataFrame) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for name in FEATURE_NAMES:
        values = df[name]
        minimum = float(values.min())
        maximum = float(values.max())
        median = float(values.median())
        span = max(maximum - minimum, 0.01)
        step = 0.001 if span < 1 else 0.01
        specs.append(
            {
                "name": name,
                "label": FEATURE_LABELS[name],
                "min": round(minimum, 4),
                "max": round(maximum, 4),
                "median": round(median, 4),
                "step": step,
            }
        )
    return specs


def _make_classifier() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=420,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def _make_regressor() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=360,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ),
        ]
    )


def train_and_save() -> dict[str, Any]:
    df = load_dataset()
    x = df[FEATURE_NAMES]
    y_binary = (df["quality"] >= 6).astype(int)
    y_quality = df["quality"].astype(float)

    (
        x_train,
        x_test,
        y_binary_train,
        y_binary_test,
        y_quality_train,
        y_quality_test,
    ) = train_test_split(
        x,
        y_binary,
        y_quality,
        test_size=0.2,
        random_state=42,
        stratify=y_binary,
    )

    classifier = _make_classifier()
    regressor = _make_regressor()
    classifier.fit(x_train, y_binary_train)
    regressor.fit(x_train, y_quality_train)

    y_pred = classifier.predict(x_test)
    y_proba = classifier.predict_proba(x_test)[:, 1]
    y_quality_pred = regressor.predict(x_test)

    rf_model = classifier.named_steps["model"]
    importances = [
        {"name": name, "label": FEATURE_LABELS[name], "importance": float(score)}
        for name, score in zip(FEATURE_NAMES, rf_model.feature_importances_)
    ]
    importances.sort(key=lambda item: item["importance"], reverse=True)

    metadata = {
        "datasetRows": int(len(df)),
        "target": "quality >= 6",
        "positiveClassShare": round(float(y_binary.mean()), 4),
        "metrics": {
            "accuracy": round(float(accuracy_score(y_binary_test, y_pred)), 4),
            "precision": round(float(precision_score(y_binary_test, y_pred)), 4),
            "recall": round(float(recall_score(y_binary_test, y_pred)), 4),
            "f1": round(float(f1_score(y_binary_test, y_pred)), 4),
            "rocAuc": round(float(roc_auc_score(y_binary_test, y_proba)), 4),
            "qualityMae": round(float(mean_absolute_error(y_quality_test, y_quality_pred)), 4),
            "qualityR2": round(float(r2_score(y_quality_test, y_quality_pred)), 4),
        },
        "featureSpecs": _feature_specs(df),
        "featureImportance": importances,
        "presets": _build_presets(df),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "regressor": regressor,
            "feature_names": FEATURE_NAMES,
            "metadata": metadata,
        },
        MODEL_PATH,
    )
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _build_presets(df: pd.DataFrame) -> list[dict[str, Any]]:
    low = df[df["quality"] < 6][FEATURE_NAMES].median()
    high = df[df["quality"] >= 6][FEATURE_NAMES].median()
    top = df[df["quality"] >= 7][FEATURE_NAMES].median()

    return [
        {
            "name": "House Median",
            "values": {key: round(float(value), 4) for key, value in df[FEATURE_NAMES].median().items()},
        },
        {
            "name": "Private Reserve",
            "values": {key: round(float(value), 4) for key, value in high.items()},
        },
        {
            "name": "Grand Cellar",
            "values": {key: round(float(value), 4) for key, value in top.fillna(high).items()},
        },
        {
            "name": "Sharp Table Red",
            "values": {key: round(float(value), 4) for key, value in low.items()},
        },
    ]


def ensure_model() -> dict[str, Any]:
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        return train_and_save()

    if DATA_PATH.stat().st_mtime > MODEL_PATH.stat().st_mtime:
        return train_and_save()

    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def load_bundle() -> dict[str, Any]:
    ensure_model()
    return joblib.load(MODEL_PATH)


def metadata() -> dict[str, Any]:
    return ensure_model()


def predict(values: dict[str, Any]) -> Prediction:
    bundle = load_bundle()
    row = _coerce_feature_row(values)
    frame = pd.DataFrame([row], columns=FEATURE_NAMES)

    probability = float(bundle["classifier"].predict_proba(frame)[0, 1])
    predicted_class = int(probability >= 0.5)
    quality_prediction = float(np.clip(bundle["regressor"].predict(frame)[0], 1, 10))

    return Prediction(
        high_quality_probability=probability,
        predicted_class=predicted_class,
        predicted_quality=quality_prediction,
        tier=_tier(probability, quality_prediction),
        notes=_tasting_notes(row, probability),
    )


def _coerce_feature_row(values: dict[str, Any]) -> dict[str, float]:
    missing = [name for name in FEATURE_NAMES if name not in values]
    if missing:
        raise ValueError(f"Missing feature values: {', '.join(missing)}")

    row: dict[str, float] = {}
    for name in FEATURE_NAMES:
        try:
            value = float(values[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        row[name] = value
    return row


def _tier(probability: float, predicted_quality: float) -> str:
    if probability >= 0.82 and predicted_quality >= 6.4:
        return "Grand Cru Signal"
    if probability >= 0.66:
        return "Private Reserve"
    if probability >= 0.5:
        return "Cellar Select"
    if probability >= 0.35:
        return "Table Vintage"
    return "Decant With Care"


def _tasting_notes(row: dict[str, float], probability: float) -> list[str]:
    notes: list[str] = []
    if row["alcohol"] >= 11.2:
        notes.append("Higher alcohol is supporting the quality signal.")
    else:
        notes.append("Lower alcohol keeps the model cautious.")

    if row["volatile acidity"] <= 0.45:
        notes.append("Volatile acidity is in a cleaner reserve range.")
    else:
        notes.append("Volatile acidity is the main pressure point.")

    if row["sulphates"] >= 0.65:
        notes.append("Sulphates add structure in this profile.")

    if probability >= 0.65:
        notes.append("The profile sits near the stronger half of the training cellar.")
    elif probability < 0.4:
        notes.append("The profile resembles lower-rated training samples.")

    return notes[:4]

