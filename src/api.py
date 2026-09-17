from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import joblib
import os
import numpy as np
from model import PulseLSTM

app = FastAPI(title="Pulse AI Inference API", description="Headless anomaly detection engine")

# --- MODEL LOADING (Runs once on startup) ---
try:
    with open("checkpoints/model_meta.json", "r") as f:
        meta = json.load(f)
    num_features = meta["num_features"]
    feature_cols = meta["feature_cols"]

    scaler = joblib.load("checkpoints/universal_scaler.pkl")
    
    device = torch.device("cpu")
    model = PulseLSTM(input_size=num_features, hidden_size=64, num_layers=2, output_size=num_features)
    model.load_state_dict(torch.load("checkpoints/universal_model.pt", map_location=device))
    model.eval()
    print("✅ Pulse AI weights and scaler loaded successfully.")
except Exception as e:
    print("⚠️ Error loading model artifacts. Train the model via the UI first.")

# --- API SCHEMA ---
class TelemetryPayload(BaseModel):
    # Expects a 2D list: 16 timesteps, each containing 'num_features' metrics
    data: list[list[float]]

# --- INFERENCE ENDPOINT ---
@app.post("/analyze")
def analyze_telemetry(payload: TelemetryPayload):
    if len(payload.data) != 16:
        raise HTTPException(status_code=400, detail="Payload must contain exactly 16 timesteps (15 context + 1 current).")
    
    # 1. Validate and Scale incoming data
    raw_data = np.array(payload.data)
    if raw_data.shape[1] != num_features:
        raise HTTPException(status_code=400, detail=f"Expected {num_features} features: {feature_cols}")
        
    scaled_data = scaler.transform(raw_data)
    
    # 2. Split into Context (t-15 to t-1) and Target (t)
    context_tensor = torch.tensor(scaled_data[:15], dtype=torch.float32).unsqueeze(0)
    actual_tensor = torch.tensor(scaled_data[15], dtype=torch.float32).unsqueeze(0)
    
    # 3. Neural Network Inference
    with torch.no_grad():
        prediction_tensor = model(context_tensor)
        # Calculate Mean Squared Error
        mse = torch.mean((prediction_tensor - actual_tensor) ** 2).item()
        
    # 4. Anomaly Logic (Threshold set to 0.003 based on our earlier tests)
    is_anomaly = bool(mse > 0.003)
    
    # 5. Format human-readable predictions
    pred_real = scaler.inverse_transform(prediction_tensor.numpy())[0]
    predicted_metrics = dict(zip(feature_cols, pred_real.tolist()))
    
    return {
        "status": "CRITICAL" if is_anomaly else "HEALTHY",
        "is_anomaly": is_anomaly,
        "anomaly_score_mse": round(mse, 6),
        "predicted_metrics": predicted_metrics
    }