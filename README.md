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

## Notes & Suggested Fixes

- README: fixed malformed code blocks and added setup/run instructions.
- `requirements.txt` is minimal — add runtime deps (Streamlit, Plotly, FastAPI/uvicorn, joblib). See `requirements.txt` suggestions below.
- `detect.py` appears to be an older/console-based inference script and references `get_dataloaders` and `pulse_model.pt` which do not match the current `dataset.py`/`train.py` naming. Consider one of:
  - Update `detect.py` to use `get_universal_dataloaders` and load `checkpoints/universal_model.pt`, or
  - Remove `detect.py` if you only use Streamlit and the FastAPI service.
- Checkpoint naming: training saves `checkpoints/universal_model.pt` while the repo contains `checkpoints/pulse_model.pt`. Standardize names to avoid confusion.

## Next steps (recommended)

1. Update `requirements.txt` to include all runtime packages and pin versions.
2. Either fix or remove `src/detect.py` (name mismatch: `get_dataloaders` → `get_universal_dataloaders`).
3. Add a simple `Makefile` or `scripts/` with common commands (`run-ui`, `run-api`, `train`) for contributors.
4. Add unit tests for `dataset.py` and a small integration test for `api.py`.

---

If you want, I can apply the `requirements.txt` updates and either update or remove `src/detect.py` now. Tell me which you'd prefer.

