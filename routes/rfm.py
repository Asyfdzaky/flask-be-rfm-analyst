print(">>> RFM ROUTES LOADED <<<")

import os
import pandas as pd
from flask import Blueprint, jsonify, request
from middlewares.auth_middleware import auth_required
from config import get_db_connection
from rfm_pipeline import basic_cleaning, compute_rfm, cap_and_log_transform

rfm_bp = Blueprint("rfm", __name__)

# =====================================================
# LOAD ENV & MODEL SEKALI (GLOBAL)
# =====================================================
MODEL_PATH = os.getenv("MODEL_PATH")
UPLOAD_DIR = os.getenv("UPLOAD_DIR")

# Init HF API
import requests
HF_API_URL = "https://jekoo-rfm.hf.space/predict"  # Adjust if URL differs
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_ROWS = 200_000  # safety limit

# =====================================================
# RFM PROCESS
# =====================================================
@rfm_bp.post("/process/<int:file_id>")
@auth_required
def process_rfm(file_id):
    conn = get_db_connection()
    cur = conn.cursor()

    # 1. Validate ownership
    cur.execute("""
        SELECT filename FROM upload_history
        WHERE id=%s AND user_id=%s
    """, (file_id, request.user["id"]))

    history = cur.fetchone()
    if not history:
        return jsonify({"message": "file not found or unauthorized"}), 404

    filepath = os.path.join(UPLOAD_DIR, history["filename"])

    if not os.path.exists(filepath):
        return jsonify({"error": "File missing on server"}), 404

    # 2. Load file (safe & limited)
    try:
        if filepath.endswith(".csv"):
            df = pd.read_csv(
                filepath,
                sep=None,
                engine="python",
                encoding="utf-8",
                usecols=lambda c: c.lower() in {
                    "customerid", "invoicedate", "invoiceno", "quantity", "unitprice"
                }
            )
        else:
            df = pd.read_excel(filepath)
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, sep=None, engine="python", encoding="latin1")
    except Exception as e:
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 400

    # 3. Guardrail: limit size
    if len(df) > MAX_ROWS:
        return jsonify({
            "error": f"File too large ({len(df)} rows). Max allowed is {MAX_ROWS}"
        }), 413

    # 4. Pipeline
    try:
        df = basic_cleaning(df)
        rfm_df = compute_rfm(df)
        rfm_proc, rfm_log, rfm_scaled_df, _ = cap_and_log_transform(rfm_df)
    except Exception as e:
        return jsonify({"error": f"RFM pipeline failed: {str(e)}"}), 500

    # 5. Predict via Hugging Face FastAPI API
    try:
        clusters = []
        
        headers = {
            "Content-Type": "application/json"
        }
        if HF_TOKEN:
             headers["Authorization"] = f"Bearer {HF_TOKEN}"

        # Use rfm_scaled_df because the model expects scaled features
        # And we map them to the input keys that HF app expects (R_log, F_log, M_log)
        # Note: HF App MUST rename them to R_log_sc, etc internally
        for idx, row in rfm_scaled_df.iterrows():
            payload = {
                "R_log": row['R_log_sc'],
                "F_log": row['F_log_sc'],
                "M_log": row['M_log_sc']
            }
            
            response = requests.post(HF_API_URL, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                clusters.append(result["cluster"])
            else:
                 raise Exception(f"HF API Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"ERROR calling HF API: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    rfm_proc["cluster"] = clusters

    # 6. Batch insert (FAST)
    insert_data = []
    for idx, row in rfm_proc.iterrows():
        cust = row.get("CustomerID", idx)
        cust_str = (
            str(int(cust))
            if pd.notna(cust) and str(cust).replace(".", "").isdigit()
            else str(cust)
        )

        insert_data.append((
            file_id,
            cust_str,
            int(row["Recency"]),
            int(row["Frequency"]),
            float(row["Monetary"]),
            int(row["cluster"])
        ))

    try:
        cur.executemany("""
            INSERT INTO rfm_results
            (file_id, customer_id, recency, frequency, monetary, cluster)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, insert_data)
        conn.commit()
    except Exception as e:
        conn.rollback()
        return jsonify({"error": f"DB insert failed: {str(e)}"}), 500
    finally:
        cur.close()
        conn.close()

    return jsonify({
        "message": "RFM processing complete",
        "total_customers": len(insert_data),
        "clusters": int(rfm_proc["cluster"].nunique())
    }), 200


# =====================================================
# GET RESULTS
# =====================================================
@rfm_bp.get("/results/<int:file_id>")
@auth_required
def rfm_results(file_id):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id FROM upload_history
        WHERE id=%s AND user_id=%s
    """, (file_id, request.user["id"]))

    if not cur.fetchone():
        return jsonify({"message": "not found or unauthorized"}), 404

    cur.execute("""
        SELECT customer_id, recency, frequency, monetary, cluster
        FROM rfm_results
        WHERE file_id=%s
    """, (file_id,))

    results = cur.fetchall()

    cur.close()
    conn.close()

    return jsonify({
        "message": "success",
        "file_id": file_id,
        "total": len(results),
        "data": results
    }), 200
