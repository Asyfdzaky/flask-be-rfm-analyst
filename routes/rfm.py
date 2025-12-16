print(">>> RFM ROUTES LOADED <<<")

import os
import pandas as pd
import joblib
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

if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    raise RuntimeError("MODEL_PATH not set or model file not found")

MODEL = joblib.load(MODEL_PATH)   # ✅ LOAD SEKALI

MAX_ROWS = 100_000  # safety limit

# =====================================================
# RFM PROCESS
# =====================================================
@rfm_bp.post("/process/<int:file_id>")
@auth_required
def process_rfm(file_id):
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)

    # 1️⃣ Validate ownership
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

    # 2️⃣ Load file (safe & limited)
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

    # 3️⃣ Guardrail: limit size
    if len(df) > MAX_ROWS:
        return jsonify({
            "error": f"File too large ({len(df)} rows). Max allowed is {MAX_ROWS}"
        }), 413

    # 4️⃣ Pipeline
    try:
        df = basic_cleaning(df)
        rfm_df = compute_rfm(df)
        rfm_proc, rfm_log, rfm_scaled_df, _ = cap_and_log_transform(rfm_df)
    except Exception as e:
        return jsonify({"error": f"RFM pipeline failed: {str(e)}"}), 500

    # 5️⃣ Predict
    try:
        clusters = MODEL.predict(rfm_scaled_df)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    rfm_proc["cluster"] = clusters

    # 6️⃣ Batch insert (FAST)
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
    cur = conn.cursor(dictionary=True)

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
