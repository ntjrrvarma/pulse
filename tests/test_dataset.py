import numpy as np
import pandas as pd
from src.dataset import UniversalTimeSeriesDataset


# Note: Added tmp_path to the function arguments (pytest provides this automatically)
def test_sliding_window_dimensions(dummy_timeseries_data, tmp_path):
    seq_length = 10
    feature_cols = ["cpu", "memory", "network"]
    
    # 1. Write the synthetic numpy data to a temporary CSV file
    df = pd.DataFrame(dummy_timeseries_data, columns=feature_cols)
    csv_file = tmp_path / "dummy_telemetry.csv"
    df.to_csv(csv_file, index=False)
    
    # 2. Pass the file path to the dataset, matching your real application flow
    dataset = UniversalTimeSeriesDataset(
        csv_path=str(csv_file),
        seq_length=seq_length,
        feature_cols=feature_cols
    )
    
    assert len(dataset) == 90
    
    # Unpack 3 values to match your dataset's __getitem__ signature
    x, y, _ = dataset[0] 
    
    assert x.shape == (10, 3)