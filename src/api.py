from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import joblib
import os
import numpy as np
from src.model import PulseLSTM

# Global runtime state
MODEL = None
SCALER = None
META = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, SCALER, META
    meta_path = "checkpoints/model_meta.json"
    weights_path = "checkpoints/universal_model.pt"
    scaler_path = "checkpoints/universal_scaler.pkl"

    if not os.path.exists(meta_path) or not os.path.exists(weights_path):
        raise RuntimeError("Missing model artifacts. Run training first.")

    with open(meta_path, "r") as f:
        META = json.load(f)

    SCALER = joblib.load(scaler_path)

    MODEL = PulseLSTM(
        input_size=META["num_features"],
        hidden_size=64,
        num_layers=2,
        output_size=META["num_features"]
    )
    MODEL.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    MODEL.eval()
    print(f"Loaded Pulse model. Features: {META['feature_cols']} | Dynamic Threshold: {META['threshold_mse']}")

    yield


app = FastAPI(
    title="Pulse AI Anomaly Service",
    description="Adaptive neural network inference microservice",
    lifespan=lifespan,
)

class TelemetryPayload(BaseModel):
    # Expects timesteps = seq_length + 1
    data: list[list[float]]

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "configured_features": META.get("feature_cols", []),
        "required_timesteps": META.get("seq_length", 15) + 1,
        "calibrated_threshold": META.get("threshold_mse")
    }

@app.post("/analyze")
def analyze_telemetry(payload: TelemetryPayload):
    seq_length = META.get("seq_length", 15)
    required_len = seq_length + 1
    if len(payload.data) != required_len:
        raise HTTPException(
            status_code=400,
            detail=f"Expected exactly {required_len} timesteps ({seq_length} historical + 1 current target)."
        )

    num_features = META.get("num_features")
    if num_features is None:
        raise HTTPException(status_code=503, detail="Model metadata is not loaded.")

    raw_data = np.array(payload.data)
    if raw_data.shape[1] != num_features:
        feature_cols = META.get("feature_cols", [])
        raise HTTPException(
            status_code=400,
            detail=f"Expected {num_features} columns matching: {feature_cols}"
        )

    # 1. Normalize
    scaled_data = SCALER.transform(raw_data)

    # 2. Slice sequence into history window and current evaluation point
    seq_len = META["seq_length"]
    context_tensor = torch.tensor(scaled_data[:seq_len], dtype=torch.float32).unsqueeze(0)
    actual_tensor = torch.tensor(scaled_data[seq_len], dtype=torch.float32).unsqueeze(0)

    # 3. Model forward pass
    with torch.no_grad():
        pred_tensor = MODEL(context_tensor)
        sample_mse = torch.mean((pred_tensor - actual_tensor) ** 2).item()

    # 4. Evaluate against calibrated statistical cutoff
    threshold = META["threshold_mse"]
    is_anomaly = bool(sample_mse > threshold)

    pred_unscaled = SCALER.inverse_transform(pred_tensor.numpy())[0]
    predictions = dict(zip(META["feature_cols"], np.round(pred_unscaled, 2).tolist()))

    return {
        "status": "CRITICAL" if is_anomaly else "HEALTHY",
        "is_anomaly": is_anomaly,
        "anomaly_score": round(sample_mse, 6),
        "threshold": round(threshold, 6),
        "predicted_metrics": predictions
    }