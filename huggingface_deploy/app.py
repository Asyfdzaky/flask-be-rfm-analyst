
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI(title="RFM Prediction API")

# Load Model
try:
    model = joblib.load("rfm_kmeans.model")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

class RFMInput(BaseModel):
    R_log: float
    F_log: float
    M_log: float

@app.get("/")
def home():
    return {"message": "RFM Prediction API is Running"}

@app.post("/predict")
def predict(data: RFMInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare Data
        # Model expects: ['R_log_sc', 'F_log_sc', 'M_log_sc']
        input_df = pd.DataFrame([[data.R_log, data.F_log, data.M_log]], 
                                columns=["R_log_sc", "F_log_sc", "M_log_sc"])
        
        # Predict
        cluster = model.predict(input_df)[0]
        return {"cluster": int(cluster)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
