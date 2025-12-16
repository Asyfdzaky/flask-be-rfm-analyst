
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn
import io
import datetime as dt

app = FastAPI(title="RFM Prediction API")

# Load Model
try:
    model = joblib.load("rfm_kmeans.model")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# ============================
# PROCESSING LOGIC (Moves from rfm_pipeline.py)
# ============================
def basic_cleaning(df):
    """Drop missing customer IDs and remove non-positive quantity/unitprice."""
    df = df.copy()
    # Convert to numeric, coercing non-numeric values to NaN
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors='coerce')
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors='coerce')

    # Drop rows where Quantity or UnitPrice is NaN (failed conversion)
    df.dropna(subset=["Quantity", "UnitPrice"], inplace=True)

    df.dropna(subset=["CustomerID"], inplace=True)
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # ensure InvoiceDate is datetime
    # Handle odd formats like "01/12/2010 08.45" where dot is used for time
    if df["InvoiceDate"].dtype == 'object':
         df["InvoiceDate"] = df["InvoiceDate"].astype(str).str.replace(".", ":", regex=False)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=True, errors='coerce')
    
    # Drop rows where InvoiceDate failed to parse
    df.dropna(subset=["InvoiceDate"], inplace=True)

    # compute Amount
    df["Amount"] = df["Quantity"] * df["UnitPrice"]

    return df

def compute_rfm(df, reference_date=None):
    """Compute RFM table per CustomerID."""
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (reference_date - x.max()).days,
        "InvoiceNo": "nunique",
        "Amount": "sum"
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Monetary"]
    return rfm

def cap_and_log_transform(rfm):
    """Cap extreme values (99th percentile), log-transform, then scale manually."""
    rfm_proc = rfm.copy()

    Q99_F = rfm_proc["Frequency"].quantile(0.99)
    Q99_M = rfm_proc["Monetary"].quantile(0.99)

    rfm_proc["Frequency_Capped"] = np.where(rfm_proc["Frequency"] > Q99_F, Q99_F, rfm_proc["Frequency"])
    rfm_proc["Monetary_Capped"] = np.where(rfm_proc["Monetary"] > Q99_M, Q99_M, rfm_proc["Monetary"])
    rfm_proc["Recency_Capped"] = rfm_proc["Recency"]

    rfm_log = pd.DataFrame({
        "R_log": np.log(rfm_proc["Recency_Capped"] + 1),
        "F_log": np.log(rfm_proc["Frequency_Capped"] + 1),
        "M_log": np.log(rfm_proc["Monetary_Capped"] + 1)
    })

    # Manual Standard Scaling
    rfm_scaled_dict = {}
    for col in ["R_log", "F_log", "M_log"]:
        mean = rfm_log[col].mean()
        std = rfm_log[col].std(ddof=0)
        if std == 0:
            rfm_scaled_dict[f"{col}_sc"] = rfm_log[col] - mean
        else:
            rfm_scaled_dict[f"{col}_sc"] = (rfm_log[col] - mean) / std

    rfm_scaled_df = pd.DataFrame(rfm_scaled_dict, index=rfm_proc.index)

    return rfm_proc, rfm_log, rfm_scaled_df


@app.get("/")
def home():
    return {"message": "RFM Processing API is Running"}

@app.post("/process_file")
async def process_file(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 1. Read File
        contents = await file.read()
        if file.filename.endswith(".csv"):
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding="utf-8")
            except:
                df = pd.read_csv(io.BytesIO(contents), encoding="latin1")
        elif file.filename.endswith((".xlsx", ".xls")):
             df = pd.read_excel(io.BytesIO(contents))
        else:
             raise HTTPException(status_code=400, detail="Invalid file format. Use CSV or Excel.")

        # 2. Pipeline
        df_clean = basic_cleaning(df)
        rfm_df = compute_rfm(df_clean)
        rfm_proc, _, rfm_scaled_df = cap_and_log_transform(rfm_df)
        
        # 3. Rename columns for Model
        # rfm_scaled_df already has R_log_sc, F_log_sc, M_log_sc
        
        # 4. Predict
        clusters = model.predict(rfm_scaled_df)
        rfm_proc["cluster"] = clusters
        
        # 5. Format Result as List of Dicts
        # Need to handle CustomerID type (float/str)
        results = []
        for idx, row in rfm_proc.iterrows():
            cust_id = row["CustomerID"]
            # Formatting similar to rfm.py logic
            if pd.notna(cust_id) and isinstance(cust_id, (float, int)):
                 cust_str = str(int(cust_id))
            else:
                 cust_str = str(cust_id)
                 
            results.append({
                "CustomerID": cust_str,
                "Recency": int(row["Recency"]),
                "Frequency": int(row["Frequency"]),
                "Monetary": float(row["Monetary"]),
                "Cluster": int(row["cluster"])
            })

        return {"results": results}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
