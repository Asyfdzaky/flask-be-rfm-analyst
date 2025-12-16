print(">>> RFM ROUTES LOADED <<<")

import os
# import pandas as pd # Removed: Processing moved to HF
from flask import Blueprint, jsonify, request
from middlewares.auth_middleware import auth_required
from config import get_db_connection
# from rfm_pipeline import basic_cleaning, compute_rfm, cap_and_log_transform # Removed: Processing moved to HF

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

    # 2. Forward File to Hugging Face
    HF_API_URL = "https://jekoo-rfm.hf.space/process_file"  # New Endpoint
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    try:
        # Prepare headers and files
        headers = {}
        if HF_TOKEN:
             headers["Authorization"] = f"Bearer {HF_TOKEN}"
        
        # Open file and stream it
        with open(filepath, 'rb') as f:
            files = {'file': (os.path.basename(filepath), f)}
            
            print("Uploading file to Hugging Face for processing...")
            response = requests.post(HF_API_URL, files=files, headers=headers, timeout=120) # 120s timeout for large files

        if response.status_code == 200:
            result_data = response.json().get("results", [])
        else:
             raise Exception(f"HF API Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"HF Processing Failed: {e}")
        return jsonify({"error": f"Processing failed at HF: {str(e)}"}), 500

    # 3. Batch insert to DB
    insert_data = []
    clusters_set = set()
    
    for row in result_data:
        insert_data.append((
            file_id,
            row["CustomerID"],
            row["Recency"],
            row["Frequency"],
            row["Monetary"],
            row["Cluster"]
        ))
        clusters_set.add(row["Cluster"])

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
        "clusters": len(clusters_set)
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
    return jsonify({
        "message": "success",
        "file_id": file_id,
        "total": len(results),
        "data": results
    }), 200

# =====================================================
# AI INSIGHTS
# =====================================================
from services.gemini_service import generate_rfm_insight

@rfm_bp.post("/insight")
@auth_required
def get_insight():
    data = request.json
    
    # Expecting: { cluster, recency, frequency, monetary, total, label }
    try:
        insight = generate_rfm_insight(
            cluster_id=data.get("cluster"),
            recency=data.get("recency"),
            frequency=data.get("frequency"),
            monetary=data.get("monetary"),
            total_customers=data.get("total"),
            segment_label=data.get("label")
        )
        
        import json
        if isinstance(insight, str):
            try:
                insight = json.loads(insight)
            except:
                pass # return as string if parse fails
                
        return jsonify({"data": insight}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
