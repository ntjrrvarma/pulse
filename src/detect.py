import torch
import numpy as np
import os
from model import PulseLSTM
from dataset import get_dataloaders

def run_inference_engine(threshold_multiplier=4.0):
    device = torch.device("cpu")
    
    # 1. Load the data (batch_size=1 to simulate a live 1-minute streaming feed)
    csv_path = os.path.join("data", "server_metrics.csv")
    _, test_loader, scaler = get_dataloaders(csv_path, seq_length=15, batch_size=1)
    
    # 2. Load the trained PyTorch weights
    model = PulseLSTM(input_size=3, hidden_size=64, num_layers=2, output_size=3).to(device)
    model.load_state_dict(torch.load(os.path.join("checkpoints", "pulse_model.pt")))
    model.eval()

    print("Pulse AI Engine loaded. Monitoring telemetry stream...\n")
    
    errors = []
    alerts_triggered = 0
    
    with torch.no_grad():
        for i, (batch_x, batch_y, actual_label) in enumerate(test_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # The AI predicts the next minute
            prediction = model(batch_x)
            
            # Calculate how wrong the AI was
            mse = torch.mean((prediction - batch_y) ** 2).item()
            errors.append(mse)
            
            # We wait 100 minutes to establish a dynamic baseline threshold
            if i > 100:
                # Dynamic threshold: Rolling mean error + (Multiplier * Standard Deviation)
                rolling_mean = np.mean(errors[-100:-1])
                rolling_std = np.std(errors[-100:-1])
                threshold = rolling_mean + (threshold_multiplier * rolling_std)
                
                if mse > threshold:
                    # Un-normalize the data so it makes sense in the console output
                    pred_real = scaler.inverse_transform(prediction.numpy())[0]
                    actual_real = scaler.inverse_transform(batch_y.numpy())[0]
                    
                    print(f"[CRITICAL ALERT] Timestamp: +{i} mins | Anomaly Score: {mse:.4f}")
                    print(f"  -> Predicted: CPU {pred_real[0]:.1f}% | Mem {pred_real[1]:.1f}% | Net {pred_real[2]:.1f} MB/s")
                    print(f"  -> Actual:    CPU {actual_real[0]:.1f}% | Mem {actual_real[1]:.1f}% | Net {actual_real[2]:.1f} MB/s")
                    print("-" * 65)
                    alerts_triggered += 1

    print(f"\nMonitoring complete. Total critical anomalies caught: {alerts_triggered}")

if __name__ == "__main__":
    run_inference_engine()