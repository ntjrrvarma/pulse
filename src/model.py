import torch
import torch.nn as nn

class PulseLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3, dropout=0.2):
        """
        Args:
            input_size: Number of features (CPU, Mem, Net = 3)
            hidden_size: Number of neurons in the hidden layers
            num_layers: Number of stacked LSTM layers
            output_size: Number of features to predict (3)
        """
        super(PulseLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # batch_first=True matches our dataset output: (Batch, Sequence, Features)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # The fully connected layer maps the LSTM's hidden state back down to our 3 metrics
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize the hidden state (h0) and cell state (c0) with zeros
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Pass the input sequence through the LSTM
        # 'out' contains the hidden states for ALL 15 timesteps
        out, _ = self.lstm(x, (h0, c0))
        
        # We only care about the prediction at the final timestep for our forecast
        last_time_step_out = out[:, -1, :]
        
        # Pass that final state through the linear layer to get the actual metric predictions
        predictions = self.fc(last_time_step_out)
        
        return predictions

if __name__ == "__main__":
    # Test the architecture with a dummy tensor matching our DataLoader shapes
    model = PulseLSTM()
    dummy_input = torch.randn(32, 15, 3)
    dummy_output = model(dummy_input)
    
    print("Model initialized successfully.")
    print(model)
    print(f"\nDummy Output Shape: {dummy_output.shape}")