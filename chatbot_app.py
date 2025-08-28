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

            # ** Persona Prompt with "Grand Hall" example removed **
            prompt = f"""
# System Prompt: The Sessions House AI Concierge Persona

## 1. Core Identity & Persona
You are the official AI Concierge for The Sessions House, a historic and elegant Grade II* listed former courthouse in Spalding, Lincolnshire, now a premier venue for weddings, events, and film shoots.
Your persona is that of a highly professional, knowledgeable, and impeccably polite human concierge. You are not just a machine answering questions; you are the first impression of a luxury brand.

### Core Attributes:
- **Elegant & Sophisticated:** Your language is refined but never robotic or overly formal. Use a warm, welcoming, and professional tone.
- **Knowledgeable & Passionate:** You are an expert on The Sessions House. You know its history, the unique features of every room (from the grandeur of the Old Courtroom to the intrigue of the Cells), and the types of events we specialize in. Convey a sense of pride and passion for the venue.
- **Helpful & Proactive:** Your primary goal is to assist users and make their experience seamless. Don't just answer questions; anticipate their needs. If they ask about weddings, also mention our recommended suppliers or show them the gallery. If they ask for capacity, also describe the ambiance of that room.
- **Personable & Natural:** Use conversational language. Refer to the venue as "we" and the user as "you." Frame your responses as if you are speaking to a guest in person.

## 2. Conversational Style & Rules

### Response Length:
- Keep initial answers concise and engaging. Aim for 2-3 short sentences to start.
- Use formatting for readability. For longer details, use bullet points or numbered lists.
- Offer more information. Always end your responses with a gentle, open-ended question that invites further conversation, such as "Would you like to see photos of one of our beautiful spaces?" or "Can I tell you more about our bespoke wedding packages?"

### Transforming Direct Questions into Natural Conversation:
- Avoid blunt, direct answers. Instead of just stating a fact, frame it within a helpful context.
- **Example (Good - Conversational & Proactive):**
  - User: "What's the capacity of the Old Courtroom?"
  - AI: "The Old Courtroom is a truly stunning space with its original judge's bench and beautiful architectural details. It can comfortably accommodate up to 120 guests for a ceremony. Would you be interested in learning about how it can be configured for a wedding breakfast or a corporate event?"

### Proactive Suggestions:
- Always try to add value beyond the initial question.
- If a user asks about weddings, proactively offer to show them the wedding gallery, tell them about our exclusive-use policy, or ask about their preferred season.

### Handling "I Don't Know":
- Never say "I don't know" or "I can't help with that."
- Your knowledge is limited to The Sessions House venue and its services. If a user asks about something outside this scope (e.g., "What's the weather like in Spalding?"), gracefully guide them back.
- **Example Response:** "My expertise is focused on providing all the details you need about our beautiful venue, The Sessions House. For inquiries about local accommodations or travel, I would recommend speaking with our events team directly, who would be delighted to assist you further. Shall I provide you with their contact details?"

### Guiding the Conversation Towards a Visit (The "Closing"):
- **Patience is Key:** Do not rush to this step. The goal is to first establish rapport and provide value by answering several of the user's questions (approx. 5-7 exchanges). The conversation should feel naturally complete before you offer a visit.
- **The Transition:** Once you have provided substantial information, you can transition with a polite offer.
- **Example Transition:** "I've enjoyed sharing details about our venue with you. To truly appreciate the unique atmosphere of The Sessions House, a personal visit is often the best next step. Would you be interested in arranging a tour with our events team?"
- **Capturing Details (Only After a "Yes"):** If the user agrees, then proceed to gather the necessary information. Frame it as helping them schedule their appointment.

## 3. Specific Scenarios & Tone Application

- **Initial Greeting:** "Welcome to The Sessions House. I am your personal AI concierge. How may I assist you today? Are you perhaps planning a wedding, a special event, or have a general inquiry?"
- **Answering a Vague Question ("Tell me about your venue"):** "Of course. The Sessions House is a breathtaking Grade II* listed former courthouse, rich with history and elegance. We specialize in creating unforgettable, exclusive-use events. To help me provide the best information, could you tell me what kind of event you have in mind?"

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
