import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# --- Initialization ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- AI Configuration ---
MODEL_CONFIGURED = False
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    MODEL_CONFIGURED = True
    print("Gemini AI Model configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini AI: {e}")
    model = None

# --- API Routes ---
@app.route("/")
def home():
    return "Hello, the AI Server is running!"

@app.route("/chat", methods=['POST'])
def chat():
    if not MODEL_CONFIGURED:
        return jsonify({"error": "AI model is not available."}), 500

    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    try:
        # Send the message directly to the generic Gemini AI
        response = model.generate_content(user_message)
        bot_response_text = response.text

        return jsonify({"response": bot_response_text})

    except Exception as e:
        print(f"Error during AI generation: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500
