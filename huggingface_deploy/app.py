
import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Load Model
model = joblib.load("rfm_kmeans.model")

def predict(R_log, F_log, M_log):
    # Prepare Data
    data = pd.DataFrame([[R_log, F_log, M_log]], columns=["R_log", "F_log", "M_log"])
    
    # Predict
    cluster = model.predict(data)[0]
    return int(cluster)

# Gradio Interface (API Mode)
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="R_log"),
        gr.Number(label="F_log"),
        gr.Number(label="M_log")
    ],
    outputs="number",
    title="RFM Cluster Prediction API",
    description="API untuk memprediksi cluster pelanggan berdasarkan nilai log RFM."
)

iface.launch(server_name="0.0.0.0", server_port=7860)
