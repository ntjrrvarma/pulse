import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from dataset import get_universal_dataloaders
from model import PulseLSTM

def train_universal_model(csv_path: str, feature_cols: list, label_col: str = None, epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load data and dynamically get the feature count
    train_loader, test_loader, num_features = get_universal_dataloaders(
        csv_path=csv_path, 
        feature_cols=feature_cols, 
        label_col=label_col,
        batch_size=64
    )

    print(f"Detected {num_features} dimensions. Building dynamic LSTM...")

    # 2. Build model with dynamic input/output sizes
    model = PulseLSTM(input_size=num_features, hidden_size=64, num_layers=2, output_size=num_features).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

    # 3. Save the model AND the metadata so the inference engine knows what it's looking at
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/universal_model.pt")
    
    metadata = {
        "num_features": num_features,
        "feature_cols": feature_cols
    }
    with open("checkpoints/model_meta.json", "w") as f:
        json.dump(metadata, f)
        
    print("\nUniversal model trained and metadata saved.")

if __name__ == "__main__":
    # You can now swap this out for ANY dataset. 
    # Stock prices: ['open', 'close', 'volume']
    # Weather: ['temperature', 'humidity', 'wind_speed']
    target_columns = ['cpu_pct', 'mem_pct', 'net_in_mbps']
    train_universal_model("data/server_metrics.csv", feature_cols=target_columns, label_col='is_anomaly', epochs=15)