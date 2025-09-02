import os
import fitz
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from flask import Flask, request, jsonify, Response, stream_with_context
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
SAFE_CHAR_LIMIT = 30000 

# --- AI, Google Sheets Config ---
try:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key: raise ValueError("GOOGLE_API_KEY not found.")
    genai.configure(api_key=api_key)
    # Using the more powerful Pro model
    model = genai.GenerativeModel('gemini-1.5-pro')
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
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, timeout=20, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' in content_type:
            with fitz.open(stream=io.BytesIO(response.content)) as doc:
                return "".join(page.get_text() for page in doc)
        elif 'text/html' in content_type:
            soup = BeautifulSoup(response.content, 'html.parser')
            for s in soup(["script", "style", "nav", "footer", "header"]): s.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk)
        else: return ""
    except requests.RequestException as e:
        print(f"--- Error fetching or reading URL {url}: {e}")
        return ""

def load_knowledge_base():
    global KNOWLEDGE_BASE_TEXT, knowledge_base_loaded
    if knowledge_base_loaded: return
    print("--- Starting knowledge base load...")
    all_text = []
    current_char_count = 0
    if os.path.isdir(KNOWLEDGE_DIR):
        for root, dirs, files in os.walk(KNOWLEDGE_DIR):
            for filename in sorted(files):
                if current_char_count >= SAFE_CHAR_LIMIT: break
                file_path = os.path.join(root, filename)
                try:
                    text = ""
                    if filename.lower().endswith('.pdf'):
                        with fitz.open(file_path) as doc: text = "".join(page.get_text() for page in doc)
                    elif filename.lower().endswith('.txt'):
                        with open(file_path, 'r', encoding='utf-8') as f: text = f.read()
                    if text:
                        all_text.append(text)
                        current_char_count += len(text)
                except Exception as e: print(f"--- Error processing file {filename}: {e}")
            if current_char_count >= SAFE_CHAR_LIMIT: break
    try:
        with open(URL_CONFIG_FILE, 'r') as f:
            urls_to_scrape = [line.strip() for line in f if line.strip()]
        for url in urls_to_scrape:
            if current_char_count >= SAFE_CHAR_LIMIT: break
            content = read_content_from_url(url)
            if content:
                all_text.append(content)
                current_char_count += len(content)
    except FileNotFoundError:
        print(f"--- Warning: URL config file '{URL_CONFIG_FILE}' not found.")
    KNOWLEDGE_BASE_TEXT = "\n\n---\n\n".join(all_text)
    if KNOWLEDGE_BASE_TEXT:
        print(f"--- Knowledge base loaded successfully with {current_char_count} characters.")
        knowledge_base_loaded = True

def log_conversation_summary(history):
    if not GSHEET_CLIENT: return
    try:
        # We will use the faster 'flash' model just for the summary to save costs
        summary_model = genai.GenerativeModel('gemini-1.5-flash')
        summary_prompt = f"""Based on the following conversation, provide a one-sentence summary and extract any potential lead information (name, contact details, event type, guest count, desired date). Conversation: {history} Your output MUST be a single, valid JSON object with the keys "summary", "contact", and "details"."""
        summary_response = summary_model.generate_content(summary_prompt)
        raw_text = summary_response.text
        json_start_index = raw_text.find('{')
        json_end_index = raw_text.rfind('}') + 1
        if json_start_index != -1 and json_end_index != -1:
            clean_json_text = raw_text[json_start_index:json_end_index]
            lead_data = json.loads(clean_json_text)
        else:
            lead_data = {"summary": "Could not summarize conversation.", "contact": "N/A", "details": "N/A"}
        sheet = GSHEET_CLIENT.open(GSHEET_NAME).sheet1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = lead_data.get('summary', 'N/A')
        contact = lead_data.get('contact', 'N/A')
        details = lead_data.get('details', 'N/A')
        row = [timestamp, summary, contact, details]
        sheet.append_row(row)
        print("--- Successfully logged conversation summary to Google Sheet.")
    except Exception as e:
        print(f"--- Error logging conversation summary to Google Sheet: {e}")

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
    
    def generate_stream():
        try:
            history_text = "\n".join([f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['text']}" for msg in chat_history])
            
            safe_knowledge_text = KNOWLEDGE_BASE_TEXT[:SAFE_CHAR_LIMIT]

            prompt = f"""
# System Prompt: The Sessions House AI Concierge Persona

## 1. Core Identity & Persona
You are the official AI Concierge for The Sessions House. Your persona is that of a highly professional, knowledgeable, and impeccably polite human concierge. Your language is refined, warm, welcoming, and professional. You are an expert on The Sessions House. Your primary goal is to inspire and assist potential clients by painting a vivid picture of what their event could be like.

## 2. Conversational Style & Rules

### Initial Interaction (First 1-2 User Messages):
- If the user mentions "wedding", your first response should be celebratory and open-ended.
- **Good Example:** "That's wonderful news, congratulations on your engagement! We'd be delighted to host your wedding. The Sessions House is a truly unique and historic venue, offering exclusive use for your special day. To help you get started, what would you most like to know?"
- Do not ask for specific details like guest count or date in the very first wedding-related response.

### Response Length & Flow:
- Keep answers concise and engaging. Aim for 2-4 short sentences.
- Your goal is a back-and-forth conversation. Do not provide a long monologue.
- Always end your responses with a gentle, open-ended question that invites the user to continue the conversation.

### Proactive Suggestions:
- If a user asks about **weddings**, proactively describe our exclusive-use policy or ask about their preferred season.
- If a user asks about **corporate events**, mention our AV capabilities or breakout room options.

### Handling User Uncertainty:
- If a user expresses that they are unsure (e.g., "not sure yet", "I don't know"), **do not repeat a similar question**.
- Acknowledge their uncertainty with empathy (e.g., "That's perfectly alright! Planning is a big process.").
- Then, pivot to a general, helpful offer, such as: "To help you get started, perhaps we could explore some of the beautiful spaces we have here?"

### Guiding the Conversation & Providing Contact Details:
- **Patience is Key:** Do not rush to ask for user details. First, establish rapport and provide value by answering several questions.
- **Providing Our Details:** If the user asks for our contact details or confirms they want them, provide the following information clearly.
  - **Email:** info@thesessionshouse.com
  - **WhatsApp:** 07340423610

## 3. Advanced Conversational Logic - MOST IMPORTANT RULES

### Rule A: Prioritize Answering Directly
- Your absolute first priority is to fully and directly answer the user's **most recent question**.
- Look at the last user message in the `Conversation History`. Your response MUST address it.
- Only after you have provided a complete answer are you allowed to ask a gentle follow-up question.

### Rule B: No Repeated Questions
- Before asking a question (like for guest count, event date, etc.), you MUST check the `Conversation History`.
- If that information has already been provided by the user in a previous message, you MUST NOT ask for it again. Acknowledge that you already have the information.
- **Example:** If the user says their guest count is 80, and later you need it again, you should say something like, "Working with the 80 guests you mentioned..." instead of asking again.

---
**Conversation History:**
{history_text}
---

**Knowledge Base Context:**
{safe_knowledge_text}
---

Based on all the instructions, history, and context, provide a helpful and conversational answer to the new user's question.

**New User Question:** {user_question}
"""

            stream = model.generate_content(
                prompt,
                stream=True,
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            full_response_text = ""
            for chunk in stream:
                if chunk.text:
                    full_response_text += chunk.text
                    yield chunk.text
            
            final_history = f"{history_text}\nAssistant: {full_response_text}"
            log_conversation_summary(final_history)

        except Exception as e:
            print(f"--- [CRITICAL] Error in /chat stream: {e}")
            yield "I'm sorry, an error occurred while I was thinking. Please try again."

    return Response(stream_with_context(generate_stream()), mimetype='text/plain')


