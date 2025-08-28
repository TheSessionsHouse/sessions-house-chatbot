from flask import Flask, jsonify
from flask_cors import CORS

# --- Initialization ---
app = Flask(__name__)
CORS(app)

# --- API Routes ---
@app.route("/")
def home():
    return "The minimal server is running correctly!"

@app.route("/chat", methods=['POST'])
def chat():
    return jsonify({"response": "The basic connection is working."})
