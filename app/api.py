from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest

from app.config import DEFAULT_MODEL_DIR
from app.model_handler import ModelRegistry
from app.schemas import ValidationError, normalize_payload


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_payload"):
            payload.update(record.extra_payload)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("credit_default_api")
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)

    return logger


def create_app(model_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    logger = configure_logging()
    registry = ModelRegistry(model_dir or os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))
    app.config["MODEL_REGISTRY"] = registry
    app.config["API_LOGGER"] = logger

    @app.get("/health")
    def health() -> tuple[Any, int]:
        status = "healthy" if registry.is_ready() else "unhealthy"
        code = 200 if registry.is_ready() else 503
        return jsonify(
            {
                "status": status,
                "service": "credit-card-default-service",
                "loaded_models": registry.loaded_versions,
            }
        ), code

    @app.post("/predict")
    def predict() -> tuple[Any, int]:
        started_at = time.perf_counter()
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        try:
            payload = request.get_json(force=True, silent=False)
            normalized = normalize_payload(payload, registry.feature_names)
            request_id = normalized["request_id"] or request_id

            result = registry.predict(
                normalized["features"],
                requested_version=normalized["model_version"],
                ab_key=normalized["ab_key"] or request_id,
            )

            response = {
                "request_id": request_id,
                **result,
            }
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            logger.info(
                "prediction_completed",
                extra={
                    "extra_payload": {
                        "event": "prediction_completed",
                        "request_id": request_id,
                        "model_version": result["model_version"],
                        "ab_group": result["ab_group"],
                        "prediction": result["prediction"],
                        "probability": result["probability"],
                        "duration_ms": duration_ms,
                    }
                },
            )
            return jsonify(response), 200

        except (BadRequest, ValidationError) as exc:
            return jsonify({"request_id": request_id, "error": str(exc)}), 400
        except Exception as exc:
            logger.exception(
                "prediction_failed",
                extra={"extra_payload": {"event": "prediction_failed", "request_id": request_id}},
            )
            return jsonify({"request_id": request_id, "error": str(exc)}), 500

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
