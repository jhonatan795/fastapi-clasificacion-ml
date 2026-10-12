"""Pruebas API con FastAPI TestClient (sin levantar servidor externo)."""
from fastapi.testclient import TestClient
from app.main import app


def test_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["model_loaded"] is True


def test_schema():
    with TestClient(app) as client:
        schema = client.get("/schema").json()
        assert len(schema["feature_names_in_order"]) == 30
        assert schema["labels"]["0"] == "maligno"


def test_prediction_and_probability():
    with TestClient(app) as client:
        example = client.get("/example").json()
        res = client.post("/predict", json=example)
        assert res.status_code == 200
        body = res.json()
        assert body["prediction"] in ("maligno", "benigno")
        assert body["class_id"] in (0, 1)
        assert 0 <= body["confidence"] <= 1
        assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-5
        assert res.headers.get("X-Request-ID")


def test_prediction_reproducibility():
    with TestClient(app) as client:
        sample = client.get("/example").json()
        a = client.post("/predict", json=sample).json()
        b = client.post("/predict", json=sample).json()
        assert a == b


def test_batch():
    with TestClient(app) as client:
        sample = client.get("/example").json()
        res = client.post("/predict/batch", json={"instances": [sample, sample]})
        assert res.status_code == 200
        assert len(res.json()) == 2


def test_incorrect_feature_count_returns_422():
    with TestClient(app) as client:
        res = client.post("/predict", json={"features": [1.0] * 29})
        assert res.status_code == 422


def test_nan_infinity_and_text_are_rejected():
    with TestClient(app) as client:
        example = client.get("/example").json()
        example["features"][0] = "texto"
        assert client.post("/predict", json=example).status_code == 422


def test_metrics_count_errors():
    with TestClient(app) as client:
        client.post("/predict", json={"features": []})
        res = client.get("/metrics")
        assert res.status_code == 200
        assert res.json()["requests_total"] >= 1
        assert res.json()["errors_total"] >= 1


def test_openapi_documentation():
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 200
        assert "/predict" in client.get("/openapi.json").json()["paths"]
