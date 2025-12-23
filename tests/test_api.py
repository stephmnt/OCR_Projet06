import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app, DATA_PATH


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _build_payload(preprocessor):
    df = pd.read_parquet(DATA_PATH, columns=preprocessor.input_feature_columns)
    required = [c for c in preprocessor.required_raw_columns if c in df.columns]
    mask = df[required].notna().all(axis=1)
    if not mask.any():
        raise AssertionError("No valid sample row found for prediction test.")
    row = df[mask].iloc[0].to_dict()
    return {"data": row}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_features(client):
    resp = client.get("/features")
    assert resp.status_code == 200
    payload = resp.json()
    assert "input_features" in payload
    assert "required_input_features" in payload
    assert "SK_ID_CURR" in payload["input_features"]
    assert "AMT_INCOME_TOTAL" in payload["input_features"]


def test_predict(client):
    preprocessor = client.app.state.preprocessor
    payload = _build_payload(preprocessor)
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    result = data["predictions"][0]
    assert "sk_id_curr" in result
    assert "prediction" in result
    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0
