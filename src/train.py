import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from dataset import get_universal_dataloaders
from model import PulseLSTM

def train_universal_model(
    csv_path: str, 
    feature_cols: list, 
    label_col: str = None, 
    seq_length: int = 15,
    epochs: int = 15, 
    learning_rate: float = 0.001,
    batch_size: int = 64
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 1. Dynamically initialize universal dataloaders
    train_loader, val_loader, num_features = get_universal_dataloaders(
        csv_path=csv_path,
        feature_cols=feature_cols,
        label_col=label_col,
        seq_length=seq_length,
        batch_size=batch_size
    )

    print(f"Dataset mapped: {num_features} features, Sequence window: {seq_length}")

    # 2. Build model architecture
    model = PulseLSTM(
        input_size=num_features, 
        hidden_size=64, 
        num_layers=2, 
        output_size=num_features
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 3. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y, _ in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.6f}")

    # 4. Calibration Phase (Compute MSE distribution over validation split)
    model.eval()
    val_errors = []
    
    with torch.no_grad():
        for batch_x, batch_y, _ in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            preds = model(batch_x)
            # Per-sample MSE
            sample_mse = torch.mean((preds - batch_y) ** 2, dim=1).cpu().numpy()
            val_errors.extend(sample_mse)

    val_errors = np.array(val_errors)
    mean_mse = float(np.mean(val_errors))
    std_mse = float(np.std(val_errors))
    p99_threshold = float(np.percentile(val_errors, 99))
    sigma3_threshold = float(mean_mse + (3 * std_mse))

    # Select the conservative threshold between 3-sigma and P99
    calibrated_threshold = max(sigma3_threshold, p99_threshold)

    print(f"\n--- Model Calibration ---")
    print(f"Validation Mean MSE:  {mean_mse:.6f}")
    print(f"Validation Std MSE:   {std_mse:.6f}")
    print(f"3-Sigma Cutoff:       {sigma3_threshold:.6f}")
    print(f"99th Percentile:      {p99_threshold:.6f}")
    print(f"Dynamic Threshold:    {calibrated_threshold:.6f}")

    # 5. Persist artifacts & metadata
    os.makedirs("checkpoints", exist_ok=True)
    model_path = os.path.join("checkpoints", "universal_model.pt")
    torch.save(model.state_dict(), model_path)

    metadata = {
        "num_features": num_features,
        "feature_cols": feature_cols,
        "seq_length": seq_length,
        "threshold_mse": calibrated_threshold,
        "mean_mse": mean_mse,
        "std_mse": std_mse,
        "p99_mse": p99_threshold
    }
    
    meta_path = os.path.join("checkpoints", "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved model to '{model_path}' and metadata to '{meta_path}'.")

if __name__ == "__main__":
    # Features excluding monotonic timestamps
    metrics = ["cpu_pct", "mem_pct", "net_in_mbps"]
    train_universal_model(
        csv_path="data/server_metrics.csv", 
        feature_cols=metrics, 
        label_col="is_anomaly", 
        seq_length=15, 
        epochs=15
    )