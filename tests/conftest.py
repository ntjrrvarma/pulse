import pytest
import torch
import numpy as np
from fastapi.testclient import TestClient
from src.api import app
from src.model import PulseLSTM
from src.dataset import UniversalTimeSeriesDataset

@pytest.fixture
def dummy_timeseries_data():
    """Generates 100 timesteps of 3-feature synthetic data."""
    return np.random.rand(100, 3)

@pytest.fixture
def api_client():
    """FastAPI test client."""
    return TestClient(app)

@pytest.fixture
def mock_lstm():
    """Returns an initialized LSTM model for 3 features."""
    # Assuming signature: PulseLSTM(input_size, hidden_size, num_layers)
    return PulseLSTM(input_size=3, hidden_size=16, num_layers=1)