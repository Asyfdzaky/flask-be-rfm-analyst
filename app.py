from flask import Flask
from flask_cors import CORS
from routes.auth import auth_bp
from routes.upload import upload_bp
from routes.rfm import rfm_bp
from routes.user import user_bp
import os
from config import UPLOAD_DIR

app = Flask(__name__)
# Allow all origins for CORS to fix frontend connection issues
CORS(app, resources={r"/*": {"origins": "*"}})

app.register_blueprint(auth_bp, url_prefix="/api/auth")
app.register_blueprint(upload_bp, url_prefix="/api")
app.register_blueprint(rfm_bp, url_prefix="/api/rfm")
app.register_blueprint(user_bp, url_prefix="/api/user")
print(app.url_map)

# ensure upload dir exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
