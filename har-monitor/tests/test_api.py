import numpy as np
from fastapi.testclient import TestClient

from app import app, get_samples

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and len(body["classes"]) == 6 and body["features"] == 561


def test_metrics():
    m = client.get("/api/metrics").json()
    assert 0.9 < m["keras"]["test_accuracy"] < 1
    assert len(m["keras"]["confusion"]) == 6


def test_predict_real_window_matches_label_mostly():
    X, y, _ = get_samples()
    classes = client.get("/api/health").json()["classes"]
    correct = 0
    for i in range(60):
        r = client.post("/api/predict", json={"features": X[i].tolist()})
        assert r.status_code == 200
        correct += r.json()["activity"] == classes[int(y[i])]
    assert correct / 60 > 0.8


def test_predict_rejects_wrong_length():
    r = client.post("/api/predict", json={"features": [0.0] * 10})
    assert r.status_code == 422


def test_batch_and_limit():
    X, _, _ = get_samples()
    r = client.post("/api/predict/batch", json={"rows": X[:50].tolist()})
    assert r.status_code == 200 and r.json()["count"] == 50
    r = client.post("/api/predict/batch", json={"rows": np.zeros((1001, 561)).tolist()})
    assert r.status_code == 422


def test_stream():
    r = client.get("/api/stream?n=5&seed=1")
    assert r.status_code == 200 and len(r.json()["windows"]) == 5


def test_unknown_api_route_is_404_and_index_served():
    assert client.get("/api/nope").status_code == 404
    r = client.get("/")
    assert r.status_code == 200 and "HAR Monitor" in r.text
