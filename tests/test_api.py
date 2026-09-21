import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src import api
from src.model import PulseLSTM


def test_health_check(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "online"
    assert "calibrated_threshold" in data
    assert "required_timesteps" in data


def test_analyze_rejects_invalid_sequence_length(api_client, monkeypatch):
    monkeypatch.setattr(api, "META", {"seq_length": 15, "num_features": 3, "feature_cols": ["cpu_pct", "mem_pct", "net_in_mbps"]})

    payload = {"data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]}

    response = api_client.post("/analyze", json=payload)

    assert response.status_code == 400
    assert "Expected exactly" in response.json()["detail"]