# ⚡ Pulse AI: Universal Neural Anomaly Engine

Pulse is a multivariate time-series anomaly detection engine built with PyTorch and Streamlit. It uses a custom Long Short-Term Memory (LSTM) neural network to learn the sequential baselines of infrastructure telemetry (or any time-series data) and flags systemic deviations in real-time using dynamic Mean Squared Error (MSE) thresholds.

## 🧠 Architecture
Unlike standard feed-forward networks, Pulse utilizes an LSTM architecture to maintain long-term context, allowing it to detect gradual state changes (like memory leaks) alongside sudden spikes (like DDoS traffic). 

The engine is completely dynamic. It decouples the data extraction from the neural network, allowing it to automatically resize its input/output tensors and train a custom `.pt` model on any uploaded CSV data.

* **Backend:** PyTorch (nn.LSTM, custom DataLoaders)
* **Frontend:** Streamlit, Plotly
* **Data Processing:** Scikit-learn (MinMaxScaler), Pandas, Numpy

## 📸 Dashboard 
*(Note: Replace this text with the image of your Streamlit UI showing the loss curve and the anomaly spike chart)*

## 🚀 Quickstart

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ntjrrvarma/Pulse.git](https://github.com/ntjrrvarma/Pulse.git)
   cd Pulse

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt

3. **Launch the interactive dashboard:**

   ```bash
   streamlit run src/app.py

4. **Train a Model:**

Upload data/server_metrics.csv (or your own time-series data).

Select the features you want the network to learn (e.g., cpu_pct, mem_pct). Note: Exclude monotonically increasing data like raw timestamps to maintain tensor normalization.

Hit Train and monitor the live loss trajectory.

