# ⚡ Pulse AI: Universal Neural Anomaly Engine

Pulse is a multivariate time-series anomaly detection engine built with PyTorch and Streamlit. It uses a small LSTM predictor to learn sequential baselines of infrastructure telemetry (or any time-series data) and flags deviations using MSE-based anomaly scoring.

## 🧠 Architecture

Pulse separates data extraction, normalization, model training and inference so the same pipeline can be used on different datasets. Key components:

- **Backend:** PyTorch (`nn.LSTM`) and custom DataLoaders
- **Frontend:** Streamlit and Plotly (simple interactive UI)
- **Preprocessing:** `scikit-learn`'s `MinMaxScaler`, `pandas`, `numpy`

## 🚀 Quickstart

1. Clone the repository:

```bash
git clone https://github.com/ntjrrvarma/Pulse.git
cd Pulse
```

2. Create and activate a virtual environment (recommended):

On Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Streamlit dashboard (interactive):

```bash
streamlit run src/app.py
```

5. Train the default example model from the command line:

```bash
python src/train.py
```

6. Run the headless inference API (if you prefer HTTP access):

From the project root you can run with `uvicorn`:

```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

The API expects a JSON body under the `/analyze` endpoint with `data` set to a 16xN array (15 context timesteps + 1 target) where `N` is the number of features used to train the model.