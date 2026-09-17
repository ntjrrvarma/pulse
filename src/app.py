import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import plotly.graph_objects as go
import time

from dataset import get_universal_dataloaders
from model import PulseLSTM

# UI Configuration
st.set_page_config(page_title="Pulse AI | Universal Anomaly Engine", layout="wide")
st.title("⚡ Pulse AI: Universal Neural Anomaly Engine")
st.markdown("Upload any time-series CSV, select your features, and train a custom LSTM network from scratch to detect anomalies.")

# --- SIDEBAR: DATA UPLOAD ---
st.sidebar.header("1. Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Time-Series CSV", type=["csv"])

if uploaded_file is not None:
    # Save the file locally so our backend can process it
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "uploaded_data.csv")
    with open(csv_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    df = pd.read_csv(csv_path)
    
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # --- SIDEBAR: CONFIGURATION ---
    st.sidebar.header("2. Model Architecture")
    
    # Dynamically populate multi-select with columns from the uploaded CSV
    all_columns = df.columns.tolist()
    feature_cols = st.sidebar.multiselect("Select Features to Learn", all_columns, default=all_columns[:3] if len(all_columns)>=3 else all_columns)
    label_col = st.sidebar.selectbox("Select Anomaly Label (Optional)", ["None"] + all_columns, index=0)
    
    epochs = st.sidebar.slider("Training Epochs", min_value=1, max_value=50, value=10)
    learning_rate = st.sidebar.number_input("Learning Rate", value=0.001, format="%.4f")
    
    if label_col == "None":
        label_col = None

    # --- MAIN UI: TRAINING LOOP ---
    if st.sidebar.button("Train Neural Network"):
        if not feature_cols:
            st.error("Please select at least one feature column.")
        else:
            st.markdown("### 🧠 Training Engine")
            
            # Placeholders for live UI updates
            progress_bar = st.progress(0)
            status_text = st.empty()
            loss_chart = st.empty()
            
            # 1. Initialize Data
            status_text.text("Building PyTorch Tensors...")
            train_loader, test_loader, num_features = get_universal_dataloaders(
                csv_path, feature_cols, label_col, batch_size=64
            )
            
            # 2. Initialize Model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = PulseLSTM(input_size=num_features, hidden_size=64, num_layers=2, output_size=num_features).to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            
            loss_history = []
            
            # 3. Live Training Loop
            status_text.text("Training network parameters...")
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
                loss_history.append(train_loss)
                
                # Update UI elements live
                progress_bar.progress((epoch + 1) / epochs)
                status_text.text(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.6f}")
                
                # Update live chart
                fig = go.Figure(data=[go.Scatter(y=loss_history, mode='lines+markers', name='Training Loss')])
                fig.update_layout(title="Mean Squared Error (MSE) Trajectory", height=300, margin=dict(l=0, r=0, t=30, b=0))
                loss_chart.plotly_chart(fig, use_container_width=True)
                
            st.success("Training Complete! Model weights updated.")
            
            # --- MAIN UI: INFERENCE & VISUALIZATION ---
            st.markdown("### 🔍 Live Inference & Anomaly Detection")
            with st.spinner("Running inference engine over dataset..."):
                model.eval()
                errors = []
                with torch.no_grad():
                    for batch_x, batch_y, _ in test_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        predictions = model(batch_x)
                        
                        # Calculate MSE per batch item
                        batch_mse = torch.mean((predictions - batch_y) ** 2, dim=1).cpu().numpy()
                        errors.extend(batch_mse)
                
                # Plot the anomaly scores
                st.write("If the blue line spikes, the neural network detected a deviation from the learned baseline.")
                inf_fig = go.Figure()
                inf_fig.add_trace(go.Scatter(y=errors, mode='lines', line=dict(color='red', width=2), name='Anomaly Score (MSE)'))
                inf_fig.update_layout(title="System Variance / Anomaly Spikes", height=400)
                st.plotly_chart(inf_fig, use_container_width=True)
else:
    st.info("👈 Upload a CSV file in the sidebar to begin.")