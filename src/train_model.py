from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR, FEATURE_NAMES, PROJECT_ROOT, TARGET_COLUMN
from src.data_utils import load_credit_card_data


RANDOM_STATE = 42


def build_models() -> dict[str, Pipeline]:
    return {
        "v1": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "v2": Pipeline(
            steps=[
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=8,
                        min_samples_leaf=20,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test, y_test) -> dict[str, float]:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "f1_default": round(float(f1_score(y_test, predictions, pos_label=1)), 4),
        "precision_default": round(float(precision_score(y_test, predictions, pos_label=1)), 4),
        "recall_default": round(float(recall_score(y_test, predictions, pos_label=1)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
    }


def train_and_save(data_path: Path, model_dir: Path) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    resolved_data_path = data_path.resolve()
    try:
        dataset_label = str(resolved_data_path.relative_to(PROJECT_ROOT))
    except ValueError:
        dataset_label = str(resolved_data_path)

    x, y = load_credit_card_data(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    metrics: dict[str, Any] = {
        "dataset": dataset_label,
        "target": TARGET_COLUMN,
        "feature_names": FEATURE_NAMES,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "test_size": len(x_test),
        "models": {},
    }

    for version, model in build_models().items():
        model.fit(x_train, y_train)
        model_metrics = evaluate_model(model, x_test, y_test)
        metrics["models"][version] = model_metrics

        bundle = {
            "version": version,
            "model": model,
            "feature_names": FEATURE_NAMES,
            "target": TARGET_COLUMN,
            "metrics": model_metrics,
            "trained_at": metrics["trained_at"],
        }
        joblib.dump(bundle, model_dir / f"model_{version}.joblib")

    metrics_path = model_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train credit-card default models.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_and_save(args.data_path, args.model_dir)
    print(json.dumps(metrics["models"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
