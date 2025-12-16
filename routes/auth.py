from flask import Blueprint, request, jsonify
from config import get_db_connection, SECRET_KEY
import bcrypt
import jwt
import datetime

auth_bp = Blueprint("auth", __name__)

@auth_bp.post("/register")
def register():
    data = request.json or {}
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not username or not email or not password:
        return jsonify({"message": "username, email, and password required"}), 400

    # hash password
    salt = bcrypt.gensalt()
    pw_hash = bcrypt.hashpw(password.encode(), salt)

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            INSERT INTO users (username, email, password_hash)
            VALUES (%s, %s, %s)
            RETURNING id
        """, (username, email, pw_hash))
        user_id = cur.fetchone()['id']
        conn.commit()
    except Exception as e:
        conn.rollback()
        return jsonify({"message": "username or email already exists", "error": str(e)}), 400
    finally:
        cur.close()
        conn.close()

    payload = {
        "user_id": user_id,
        "username": username,
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

    return jsonify({"message": "registered", 
                    "user_id": user_id,
                    "token": token}), 201

@auth_bp.post("/login")
def login():
    data = request.json or {}
    identifier = data.get("identifier")  # bisa username atau email
    password = data.get("password")

    if not identifier or not password:
        return jsonify({"message": "identifier and password required"}), 400

    conn = get_db_connection()
    cur = conn.cursor()

    # Cek apakah identifier cocok ke username atau email
    cur.execute("""
        SELECT id, username, email, password_hash 
        FROM users 
        WHERE username = %s OR email = %s
        LIMIT 1
    """, (identifier, identifier))

    user = cur.fetchone()

    cur.close()
    conn.close()

    if not user:
        return jsonify({"message": "invalid credentials"}), 401

    stored_hash = bytes(user["password_hash"])

    if not bcrypt.checkpw(password.encode(), stored_hash):
        return jsonify({"message": "invalid credentials"}), 401

    payload = {
        "user_id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    }

    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

    return jsonify({
        "message": "login successful",
        "token": token
    }), 200
