import os
import fitz # PyMuPDF
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# --- Initialization ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# --- Configuration ---
KNOWLEDGE_DIR = "knowledge"

# --- Global Variables ---
KNOWLEDGE_BASE_TEXT = ""
MODEL_CONFIGURED = False
knowledge_base_loaded = False

# --- AI Configuration ---
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    MODEL_CONFIGURED = True
    print("Gemini AI Model configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini AI: {e}")
    model = None

# --- Helper Function ---
def load_knowledge_base():
    """Scans the knowledge directory for PDFs and TXT files and loads them."""
    global KNOWLEDGE_BASE_TEXT, knowledge_base_loaded
    if knowledge_base_loaded: return
    print("Starting knowledge base load from local files...")

    all_text = []
    if os.path.isdir(KNOWLEDGE_DIR):
        for filename in os.listdir(KNOWLEDGE_DIR):
            file_path = os.path.join(KNOWLEDGE_DIR, filename)
            try:
                text = ""
                if filename.lower().endswith('.pdf'):
                    with fitz.open(file_path) as doc:
                        text = "".join(page.get_text() for page in doc)
                    print(f"Extracted text from PDF: {filename}")
                elif filename.lower().endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    print(f"Read text file: {filename}")

                if text:
                    all_text.append(text)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    else:
        print(f"Warning: Knowledge directory '{KNOWLEDGE_DIR}' not found.")

    KNOWLEDGE_BASE_TEXT = "\n\n---\n\n".join(all_text)
    if KNOWLEDGE_BASE_TEXT:
        print("Knowledge base loaded successfully from files.")
        knowledge_base_loaded = True
    else:
        print("Warning: Knowledge base is empty or no files could be read.")

# --- API Routes ---
@app.route("/")
def home():
    return "Hello, the AI Server with File Knowledge is running!"

@app.route("/chat", methods=['POST'])
def chat():
    if not knowledge_base_loaded:
        load_knowledge_base()

    if not MODEL_CONFIGURED:
        return jsonify({"error": "AI model is not available."}), 500

    data = request.json
    user_message = data.get('message')

    if not user_message:
        return jsonify({"error": "No message provided."}), 400

    try:
        prompt = f"""
        You are a helpful assistant for The Sessions House.
        Answer the user's question based ONLY on the information provided in the context below.
        If the information is not in the context, say you don't have that information.

        Context:
        ---
        {KNOWLEDGE_BASE_TEXT}
        ---

        User's Question: {user_message}
        """
        response = model.generate_content(prompt)
        bot_response_text = response.text.strip()

        return jsonify({"response": bot_response_text})

    except Exception as e:
        print(f"Error during AI generation: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500
        print(f"Error during AI generation: {e}")
        return jsonify({"error": "An error occurred while processing your request."}), 500
