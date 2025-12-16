import os
from flask import Blueprint, request, jsonify
from middlewares.auth_middleware import auth_required
from werkzeug.utils import secure_filename
from config  import get_db_connection

upload_bp = Blueprint("upload", __name__)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@upload_bp.post("/upload")
@auth_required
def upload_file():
    if "file" not in request.files:
        return jsonify({"message": "file is required"}), 400

    file = request.files["file"]

    if not file.filename.endswith((".csv", ".xlsx")):
        return jsonify({"message": "file must be CSV or XLSX"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    # simpan ke upload_history
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO upload_history (user_id, filename)
        VALUES (%s, %s)
        RETURNING id
    """, (request.user["id"], filename))

    upload_id = cur.fetchone()['id']
    conn.commit()

    cur.close()
    conn.close()

    return jsonify({
        "message": "file uploaded",
        "upload_id": upload_id,
        "filename": filename
    }), 201

@upload_bp.get("/history")
@auth_required
def upload_history():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT id, filename, uploaded_at
        FROM upload_history
        WHERE user_id=%s
        ORDER BY uploaded_at DESC
    """, (request.user["id"],))

    data = cur.fetchall()

    cur.close()
    conn.close()

    return jsonify(data)
