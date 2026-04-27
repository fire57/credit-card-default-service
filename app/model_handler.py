from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.config import FEATURE_NAMES, MODEL_FILENAMES


@dataclass(frozen=True)
class ModelSelection:
    version: str
    ab_group: str
    selection_method: str


class ModelRegistry:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        self._models: dict[str, dict[str, Any]] = {}
        self.load_all()

    @property
    def loaded_versions(self) -> list[str]:
        return sorted(self._models.keys())

    @property
    def feature_names(self) -> list[str]:
        if self._models:
            first_model = next(iter(self._models.values()))
            return list(first_model.get("feature_names", FEATURE_NAMES))
        return FEATURE_NAMES

    def load_all(self) -> None:
        self._models.clear()
        for version, filename in MODEL_FILENAMES.items():
            path = self.model_dir / filename
            if path.exists():
                bundle = joblib.load(path)
                self._models[version] = bundle

    def is_ready(self) -> bool:
        return bool(self._models)

    def select_model(self, requested_version: str | None, ab_key: str | None = None) -> ModelSelection:
        if requested_version:
            if requested_version not in self._models:
                raise ValueError(f"Model version '{requested_version}' is not loaded.")
            group = "control" if requested_version == "v1" else "treatment"
            return ModelSelection(requested_version, group, "explicit")

        if not {"v1", "v2"}.issubset(self._models.keys()):
            fallback_version = self.loaded_versions[0]
            group = "control" if fallback_version == "v1" else "treatment"
            return ModelSelection(fallback_version, group, "fallback_loaded_model")

        if ab_key:
            digest = hashlib.sha256(ab_key.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % 100
            version = "v1" if bucket < 50 else "v2"
            return ModelSelection(
                version=version,
                ab_group="control" if version == "v1" else "treatment",
                selection_method="hash_50_50",
            )

        version = random.choice(["v1", "v2"])
        return ModelSelection(
            version=version,
            ab_group="control" if version == "v1" else "treatment",
            selection_method="random_50_50",
        )

    def predict(
        self,
        features: dict[str, float],
        requested_version: str | None = None,
        ab_key: str | None = None,
    ) -> dict[str, Any]:
        if not self._models:
            raise RuntimeError("No models are loaded. Train models before starting the service.")

        selection = self.select_model(requested_version, ab_key)
        bundle = self._models[selection.version]
        feature_names = list(bundle.get("feature_names", FEATURE_NAMES))
        frame = pd.DataFrame([[features[name] for name in feature_names]], columns=feature_names)

        model = bundle["model"]
        prediction = int(model.predict(frame)[0])

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(frame)[0][1])
        else:
            probability = float(prediction)

        return {
            "prediction": prediction,
            "probability": probability,
            "model_version": selection.version,
            "ab_group": selection.ab_group,
            "selection_method": selection.selection_method,
        }
