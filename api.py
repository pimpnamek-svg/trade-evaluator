import subprocess
from flask import Blueprint, jsonify

api_bp = Blueprint("api", __name__)

@api_bp.route("/start-scanner", methods=["POST"])
def start_scanner():
    subprocess.Popen([
        "python",
        "scanner.py",
        "--use-websocket",
        "true",
        "--composite-scoring",
        "true"
    ])
    return jsonify({"status": "Scanner launched"})

