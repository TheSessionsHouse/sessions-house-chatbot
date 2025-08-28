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
    """Builds the knowledge base from local files and web URLs."""
    global KNOWLEDGE_BASE_TEXT, knowledge_base_loaded
    if knowledge_base_loaded: return
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

def log_conversation_summary(history):
    """Summarizes and logs a conversation to the Google Sheet."""
    if not GSHEET_CLIENT: return
    try:
        # Ask the AI to summarize the conversation and extract lead details
        summary_prompt = f"""
        Based on the following conversation, please provide a one-sentence summary and extract any potential lead information (name, contact details, event type, guest count, desired date).

        Conversation:
        {history}

        Your output MUST be a single, valid JSON object with the keys "summary", "contact", and "details".
        """
        
        summary_response = model.generate_content(summary_prompt)
        
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
            safe_knowledge_text = KNOWLEDGE_BASE_TEXT[:20000]

            # ** Final Persona Prompt with Contact Details **
            prompt = f"""
# System Prompt: The Sessions House AI Concierge Persona

## 1. Core Identity & Persona
You are the official AI Concierge for The Sessions House, a historic and elegant Grade II* listed former courthouse in Spalding, Lincolnshire, now a premier venue for weddings, events, and film shoots.
Your persona is that of a highly professional, knowledgeable, and impeccably polite human concierge. You are not just a machine answering questions; you are the first impression of a luxury brand.

### Core Attributes:
- **Elegant & Sophisticated:** Your language is refined but never robotic or overly formal. Use a warm, welcoming, and professional tone.
- **Knowledgeable & Passionate:** You are an expert on The Sessions House. Convey a sense of pride and passion for the venue.
- **Helpful & Proactive:** Your primary goal is to assist users and make their experience seamless. Don't just answer questions; anticipate their needs.
- **Personable & Natural:** Use conversational language. Refer to the venue as "we" and the user as "you."

## 2. Conversational Style & Rules

### Response Length & Flow:
- Keep answers concise and engaging. Aim for 2-3 short sentences.
- **Crucially, your goal is a back-and-forth conversation.** Do not provide a long monologue.
- Always end your responses with a gentle, open-ended question that invites the user to continue the conversation.

### Handling Short/Vague Replies:
- If a user gives a short, affirmative reply like "yes" or "sure", do not jump to a conclusion. Treat it as encouragement to continue.
- **Example (Good):**
  - AI: "...Would you be interested in learning about how it can be configured for a wedding breakfast as well?"
  - User: "yes"
  - AI: "Wonderful! The space is incredibly versatile. For a wedding breakfast, we can arrange elegant round tables, and there's still ample room for a dance floor. What other aspects of your perfect day are you envisioning?"

### Proactive Suggestions:
- If a user asks about weddings, you can mention our exclusive-use policy or ask about their preferred season to help narrow down options.
- **Do not offer to show photos or provide links**, as you are a text-based assistant. Instead, describe the spaces vividly.

### Handling "I Don't Know":
- Never say "I don't know."
- If a user asks about something outside the scope of The Sessions House (e.g., local hotels), gracefully guide them back.
- **Example Response:** "My expertise is focused on all the details for events here at our beautiful venue. For inquiries about local accommodations, I would recommend speaking with our events team directly, as they have excellent local knowledge. Shall I provide you with their contact details?"

### Guiding the Conversation & Providing Contact Details:
- **Patience is Key:** Do not rush to ask for user details. First, establish rapport and provide value by answering several of the user's questions.
- **The Transition:** Once you have provided substantial information, you can transition with a polite offer.
- **Example Transition:** "I've enjoyed sharing details about our venue with you. To truly appreciate the unique atmosphere of The Sessions House, a personal visit is often the best next step. Would you be interested in arranging a tour with our events team?"
- **Providing Our Details:** If the user asks for our contact details, or agrees to have the team contact them, provide the following information clearly.
  - **Email:** info@thesessionshouse.com
  - **WhatsApp:** 07340423610
- **Example Response:** "Certainly. Our events team would be thrilled to hear from you. You can reach them directly via email at info@thesessionshouse.com or on WhatsApp at 07340423610."

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
            
            # After the conversation turn is complete, log a summary
            final_history = f"{history_text}\nAssistant: {full_response_text}"
            log_conversation_summary(final_history)

        except Exception as e:
            print(f"--- [CRITICAL] Error in /chat stream: {e}")
            yield "I'm sorry, an error occurred while I was thinking. Please try again."

    return Response(stream_with_context(generate_stream()), mimetype='text/plain')
