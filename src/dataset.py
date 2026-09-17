import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class UniversalTimeSeriesDataset(Dataset):
    def __init__(self, csv_path: str, feature_cols: list, label_col: str = None, seq_length: int = 15):
        self.seq_length = seq_length
        self.feature_cols = feature_cols
        self.df = pd.read_csv(csv_path)
        
        # Dynamically extract however many features you passed in
        self.data = self.df[self.feature_cols].values
        self.num_features = len(self.feature_cols)
        
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(self.data)
        
        # Save the scaler so the inference engine can un-normalize later
        os.makedirs("checkpoints", exist_ok=True)
        joblib.dump(self.scaler, "checkpoints/universal_scaler.pkl")
        
        # Handle optional labels (for unsupervised datasets where we don't have anomalies mapped yet)
        if label_col and label_col in self.df.columns:
            self.labels = self.df[label_col].values
        else:
            self.labels = np.zeros(len(self.df))
            
        self.x, self.y, self.anomaly_labels = self._create_sequences()

    def _create_sequences(self):
        x_seq, y_seq, label_seq = [], [], []
        for i in range(len(self.scaled_data) - self.seq_length):
            x_seq.append(self.scaled_data[i : i + self.seq_length])
            y_seq.append(self.scaled_data[i + self.seq_length])
            label_seq.append(self.labels[i + self.seq_length])
            
        return torch.tensor(np.array(x_seq), dtype=torch.float32), \
               torch.tensor(np.array(y_seq), dtype=torch.float32), \
               torch.tensor(np.array(label_seq), dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.anomaly_labels[idx]

def get_universal_dataloaders(csv_path: str, feature_cols: list, label_col: str = None, seq_length: int = 15, batch_size: int = 64, train_split: float = 0.8):
    dataset = UniversalTimeSeriesDataset(csv_path, feature_cols, label_col, seq_length)
    
    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size
    
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, dataset.num_features