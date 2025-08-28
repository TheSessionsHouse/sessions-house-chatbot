import os
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import time
import gspread
from google.oauth2.service_account import Credentials
import json
from datetime import datetime
import io

# --- Initialization & Config ---
load_dotenv()
app = Flask(__name__)
CORS(app)
KNOWLEDGE_DIR = "knowledge"
URL_CONFIG_FILE = "urls_to_scrape.txt"
GSHEET_NAME = "Chatbot Conversation Logs"

# --- Global Variables & Setups ---
KNOWLEDGE_BASE_TEXT = ""
MODEL_CONFIGURED = False
GSHEET_CLIENT = None
knowledge_base_loaded = False

# --- AI, Google Sheets Config ---
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    MODEL_CONFIGURED = True
    print("--- Gemini AI Model configured successfully.")
except Exception as e:
    print(f"--- Error configuring Gemini AI: {e}")
    model = None
try:
    creds_json_str = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not creds_json_str: raise ValueError("GOOGLE_CREDENTIALS_JSON not found.")
    creds_info = json.loads(creds_json_str)
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
    GSHEET_CLIENT = gspread.authorize(creds)
    print("--- Google Sheets client configured successfully.")
except Exception as e:
    print(f"--- Error configuring Google Sheets client: {e}")

# --- Helper Functions ---
def read_content_from_url(url):
    """Fetches content from a URL and intelligently handles HTML vs PDF."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' in content_type:
            print(f"--- Reading PDF from URL: {url}")
            with fitz.open(stream=io.BytesIO(response.content)) as doc:
                return "".join(page.get_text() for page in doc)
        elif 'text/html' in content_type:
            print(f"--- Scraping HTML from URL: {url}")
            soup = BeautifulSoup(response.content, 'html.parser')
            for s in soup(["script", "style", "nav", "footer", "header"]): s.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk)
        else:
            print(f"--- Skipping unsupported content type '{content_type}' at URL: {url}")
            return ""
    except requests.RequestException as e:
        print(f"--- Error fetching or reading URL {url}: {e}")
        return ""

def load_knowledge_base():
    """Builds the knowledge base from local files and web URLs."""
    global KNOWLEDGE_BASE_TEXT, knowledge_base_loaded
    if knowledge_base_loaded: return
    print("--- Starting knowledge base load...")
    all_text = []
    if os.path.isdir(KNOWLEDGE_DIR):
        for filename in os.listdir(KNOWLEDGE_DIR):
            file_path = os.path.join(KNOWLEDGE_DIR, filename)
            try:
                text = ""
                if filename.lower().endswith('.pdf'):
                    with fitz.open(file_path) as doc: text = "".join(page.get_text() for page in doc)
                elif filename.lower().endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
                if text: all_text.append(text)
            except Exception as e: print(f"--- Error processing file {filename}: {e}")
    try:
        with open(URL_CONFIG_FILE, 'r') as f:
            urls_to_scrape = [line.strip() for line in f if line.strip()]
        for url in urls_to_scrape:
            content = read_content_from_url(url)
            if content: all_text.append(content)
    except FileNotFoundError:
        print(f"--- Warning: URL config file '{URL_CONFIG_FILE}' not found.")
    KNOWLEDGE_BASE_TEXT = "\n\n---\n\n".join(all_text)
    if KNOWLEDGE_BASE_TEXT:
        print("--- Knowledge base loaded successfully.")
        knowledge_base_loaded = True

def log_lead_to_sheet(lead_data):
    """Logs structured lead data to the Google Sheet."""
    if not GSHEET_CLIENT or not isinstance(lead_data, dict): return
    try:
        sheet = GSHEET_CLIENT.open(GSHEET_NAME).sheet1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = lead_data.get('summary', 'N/A')
        contact = lead_data.get('contact', 'N/A')
        details = f"Event: {lead_data.get('event_type', 'N/A')}, Guests: {lead_data.get('guest_count', 'N/A')}, Date: {lead_data.get('event_date', 'N/A')}"
        row = [timestamp, summary, contact, details]
        sheet.append_row(row)
        print("--- Successfully logged lead to Google Sheet.")
    except Exception as e:
        print(f"--- Error logging lead to Google Sheet: {e}")

# --- API Routes ---
@app.route("/")
def home():
    return "Hello, the Chatbot AI Server is fully operational!"

@app.route("/chat", methods=['POST'])
def chat():
    if not knowledge_base_loaded:
        load_knowledge_base()
    if not MODEL_CONFIGURED: return jsonify({"error": "AI model not available."}), 500
    data = request.json
    user_question = data.get('message')
    chat_history = data.get('history', [])
    if not user_question: return jsonify({"error": "No message provided."}), 400
    try:
        history_text = "\n".join([f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['text']}" for msg in chat_history])
        
        # ** THE FIX IS HERE: Truncate the knowledge base to a safe size **
        # A safe character limit to avoid API errors. 32k is a common token limit.
        # We'll use 20000 characters to be very safe.
        safe_knowledge_text = KNOWLEDGE_BASE_TEXT[:20000]

        prompt = f"""You are a proactive, inquisitive, and warm concierge for The Sessions House, a truly special and historic venue in Spalding. Your primary goal is to answer user questions based on the provided Context, and your secondary goal is to identify potential leads and capture their information.

INSTRUCTIONS:
1.  Engage the user warmly and answer their questions using ONLY the provided 'Knowledge Base Context'.
2.  Listen for signals that the user is a potential lead (e.g., asking about pricing, availability, bookings).
3.  If you identify a lead, proactively ask for their name, email or phone number, type of event, number of guests, and desired date.
4.  Your FINAL output MUST be a single, valid JSON object. This object must contain two keys: "chat_response" and "lead_data".
5.  The "chat_response" key's value must be a friendly, conversational string to show to the user.
6.  The "lead_data" key's value must be a JSON object containing the data you have collected. If you have not collected any data in this turn, the values should be null.

EXAMPLE OUTPUT FORMAT:
{{
  "chat_response": "That sounds absolutely wonderful, Jane! I've made a note of that. I'll have our wedding coordinator get in touch with you shortly at jane@example.com.",
  "lead_data": {{
    "summary": "Wedding enquiry from Jane Doe",
    "contact": "jane@example.com",
    "event_type": "Wedding",
    "guest_count": "85",
    "event_date": "October 2026"
  }}
}}

Conversation History:
---
{history_text}
---

Knowledge Base Context:
---
{safe_knowledge_text}
---

Based on all the instructions, history, and context, process the new user question below and generate your JSON output.

New User Question: {user_question}
"""

        ai_response = model.generate_content(
            prompt,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        print(f"--- Raw AI Response Text: {ai_response.text}")

        response_data = json.loads(ai_response.text)
        chat_response_for_user = response_data.get("chat_response", "I'm sorry, I'm having trouble forming a response right now.")
        lead_data_to_log = response_data.get("lead_data")

        if lead_data_to_log and any(lead_data_to_log.values()):
            log_lead_to_sheet(lead_data_to_log)
        
        return jsonify({"response": chat_response_for_user})

    except Exception as e:
        print(f"--- [CRITICAL] Error in /chat route: {e}")
        return jsonify({"error": "An error occurred while I was thinking. Please try again."}), 500
