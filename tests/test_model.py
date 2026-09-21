import torch

def test_lstm_forward_pass(mock_lstm):
    batch_size = 8
    seq_length = 10
    num_features = 3
    
    # Create dummy batch (Batch, Sequence, Features)
    dummy_input = torch.rand(batch_size, seq_length, num_features)
    
    mock_lstm.eval()
    with torch.no_grad():
        output = mock_lstm(dummy_input)
        
    # The model should predict the next timestep for all features
    assert output.shape == (batch_size, num_features)
    assert not torch.isnan(output).any(), "Model output contains NaNs"