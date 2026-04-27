import pytest

from app.api import create_app
from app.config import FEATURE_NAMES


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def sample_features():
    return {
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": -1,
        "PAY_4": -1,
        "PAY_5": -2,
        "PAY_6": -2,
        "BILL_AMT1": 3913,
        "BILL_AMT2": 3102,
        "BILL_AMT3": 689,
        "BILL_AMT4": 0,
        "BILL_AMT5": 0,
        "BILL_AMT6": 0,
        "PAY_AMT1": 0,
        "PAY_AMT2": 689,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0,
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert set(data["loaded_models"]) >= {"v1", "v2"}


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_predict_explicit_model_version(client, version):
    response = client.post(
        "/predict",
        json={"features": sample_features(), "model_version": version, "request_id": f"test-{version}"},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["model_version"] == version
    assert data["prediction"] in [0, 1]
    assert 0 <= data["probability"] <= 1


def test_predict_hash_ab_assignment(client):
    response = client.post(
        "/predict",
        json={"features": sample_features(), "ab_key": "customer-123"},
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["model_version"] in ["v1", "v2"]
    assert data["selection_method"] == "hash_50_50"


def test_predict_rejects_missing_feature(client):
    features = sample_features()
    features.pop(FEATURE_NAMES[0])

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 400
    assert "Missing required features" in response.get_json()["error"]
