from __future__ import annotations

from typing import Any


class ValidationError(ValueError):
    """Raised when the API payload cannot be converted into model features."""


def normalize_payload(payload: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError("JSON body must be an object.")

    raw_features = payload.get("features", payload)
    if not isinstance(raw_features, dict):
        raise ValidationError("'features' must be an object with client attributes.")

    missing = [feature for feature in feature_names if feature not in raw_features]
    if missing:
        raise ValidationError(f"Missing required features: {', '.join(missing)}")

    values: dict[str, float] = {}
    invalid: list[str] = []
    for feature in feature_names:
        value = raw_features[feature]
        try:
            values[feature] = float(value)
        except (TypeError, ValueError):
            invalid.append(feature)

    if invalid:
        raise ValidationError(f"Features must be numeric: {', '.join(invalid)}")

    requested_model = payload.get("model_version")
    if requested_model is not None and requested_model not in {"v1", "v2"}:
        raise ValidationError("model_version must be either 'v1' or 'v2'.")

    request_id = payload.get("request_id")
    ab_key = payload.get("ab_key") or payload.get("customer_id") or request_id

    return {
        "features": values,
        "model_version": requested_model,
        "request_id": str(request_id) if request_id is not None else None,
        "ab_key": str(ab_key) if ab_key is not None else None,
    }
