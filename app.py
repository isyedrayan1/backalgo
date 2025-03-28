print("Starting AlgoAI")
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import time
from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import re
import json
import uuid
import certifi
import random

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:8080", "http://127.0.0.1:8080", "http://localhost:8081", "http://127.0.0.1:5500", "https://backalgo-frontend.onrender.com"]}})

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get("HF_API_KEY")
if not API_KEY:
    print("Warning: HF_API_KEY not found! Please set it in environment variables.")
else:
    print(f"Loaded API Key: {API_KEY[:4]}... (hidden for security)")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
DATABASE_URL = os.environ.get("DATABASE_URL")  # Provided by Render PostgreSQL
MAX_HISTORY = 6

# SQLAlchemy setup for PostgreSQL
engine = create_engine(DATABASE_URL, echo=False)
Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_history"
    chat_id = Column(String, primary_key=True)
    user_msg = Column(Text)
    ai_msg = Column(Text)
    timestamp = Column(String)
    title = Column(String)
    welcome_shown = Column(Integer, default=0)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Database functions
def store_chat(chat_id, user_msg, ai_msg, title=None, welcome_shown=0):
    session = Session()
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        title = title or (f"Chat about {user_msg[:20]}..." if user_msg else f"Chat {chat_id}")
        ai_msg = re.sub(r'\[INST\].*?\[/INST\]', '', ai_msg, flags=re.DOTALL)
        ai_msg = re.sub(r'<s>', '', ai_msg).strip()
        chat = ChatHistory(
            chat_id=chat_id,
            user_msg=user_msg,
            ai_msg=ai_msg,
            timestamp=timestamp,
            title=title,
            welcome_shown=welcome_shown
        )
        session.merge(chat)  # Upsert behavior
        session.commit()
        print(f"Stored chat: chat_id={chat_id}, title={title}")
    except Exception as e:
        print(f"Database error during store_chat: {e}")
        session.rollback()
    finally:
        session.close()

def get_chat_history(chat_id):
    session = Session()
    try:
        chats = session.query(ChatHistory).filter_by(chat_id=chat_id).order_by(ChatHistory.timestamp).all()
        history = [{"user": chat.user_msg, "ai": re.sub(r'^You are AlgoAI.*$', '', chat.ai_msg, flags=re.MULTILINE).strip()}
                   for chat in chats if chat.user_msg or chat.ai_msg]
        return history
    except Exception as e:
        print(f"Database error during get_chat_history: {e}")
        return []
    finally:
        session.close()

def get_previous_response(chat_id):
    session = Session()
    try:
        last_chat = session.query(ChatHistory).filter_by(chat_id=chat_id).order_by(ChatHistory.timestamp.desc()).first()
        return last_chat.ai_msg if last_chat else None
    except Exception as e:
        print(f"Database error during get_previous_response: {e}")
        return None
    finally:
        session.close()

def has_welcome_been_shown(chat_id):
    session = Session()
    try:
        chat = session.query(ChatHistory).filter_by(chat_id=chat_id).first()
        return chat.welcome_shown if chat else 0
    except Exception as e:
        print(f"Database error during has_welcome_been_shown: {e}")
        return 0
    finally:
        session.close()

def get_chat_by_title_or_id(identifier):
    session = Session()
    try:
        chat = session.query(ChatHistory).filter_by(chat_id=identifier).order_by(ChatHistory.timestamp).first()
        if not chat:
            chat = session.query(ChatHistory).filter_by(title=identifier).order_by(ChatHistory.timestamp).first()
        if chat:
            history = [{"user": chat.user_msg, "ai": chat.ai_msg}] if chat.user_msg or chat.ai_msg else []
            return {"chat_id": chat.chat_id, "title": chat.title or f"Chat about {chat.user_msg[:20]}..." if chat.user_msg else f"Chat {chat.chat_id}", "history": history}
        return None
    except Exception as e:
        print(f"Database error during get_chat_by_title_or_id: {e}")
        return None
    finally:
        session.close()

def get_all_chats():
    session = Session()
    try:
        chats = session.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).all()
        unique_titles = []
        seen_chat_ids = set()
        for chat in chats:
            if chat.chat_id in seen_chat_ids:
                continue
            final_title = (chat.title if chat.title and chat.title.strip() else
                           (f"Chat about {chat.user_msg[:20]}..." if chat.user_msg and chat.user_msg.strip() else
                            f"Chat started with {chat.ai_msg[:20]}..." if chat.ai_msg and chat.ai_msg.strip() else
                            f"Chat {chat.chat_id}"))[:100]
            unique_titles.append({"chat_id": chat.chat_id, "title": final_title})
            seen_chat_ids.add(chat.chat_id)
        return unique_titles
    except Exception as e:
        print(f"Database error during get_all_chats: {e}")
        return []
    finally:
        session.close()

def get_chat_title(chat_id):
    session = Session()
    try:
        chat = session.query(ChatHistory).filter_by(chat_id=chat_id).first()
        return chat.title if chat else None
    except Exception as e:
        print(f"Database error during get_chat_title: {e}")
        return None
    finally:
        session.close()

# Rest of your functions (unchanged)
def classify_query(prompt):
    tech_keywords = r"\b(code|python|java|c\+\+|javascript|js|typescript|ruby|php|go|rust|kotlin|swift|c#|perl|scala|r|matlab|sql|nosql|algorithm|O\(.*\)|recursion|data structure|machine learning|neural network|database|API|backend|frontend|AI|time complexity|sorting|engineering|system design|software|hardware|math|algebra|calculus|geometry|statistics|probability|optimization|cloud|devops|docker|kubernetes|git|aws|azure|gcp|ci|cd|cybersecurity|game|development|network|array)\b"
    tech_patterns = [
        r"how does .* work\??", r"how to .*", r"what is the best way to .*",
        r"compare .* vs .*", r"why is .* better than .*", r"how can .* be improved\??",
        r"build .*", r"create .*", r"implement .*", r"design .*", r"optimize .*"
    ]
    greetings = ["hi", "hello", "hey", "howdy", "greetings", "salutations"]
    if prompt.strip().lower() in greetings:
        return "greeting"
    if re.search(tech_keywords, prompt, re.IGNORECASE) or any(re.search(p, prompt, re.IGNORECASE) for p in tech_patterns):
        return "tech"
    return "general"

def format_response(response):
    response = re.sub(r'\[INST\].*?\[/INST\]', '', response, flags=re.DOTALL)
    response = re.sub(r'<s>', '', response)
    response = response.strip()
    if not response:
        return "Error: No response generated. Please try again."
    response = re.sub(r'\n\s*\n{2,}', '\n\n', response, flags=re.DOTALL)
    response = re.sub(r'^\s*-\s*', '- ', response, flags=re.MULTILINE)
    if "```" in response:
        open_blocks = len(re.findall(r'```[a-zA-Z]*', response))
        close_blocks = len(re.findall(r'```', response)) - open_blocks
        if open_blocks > close_blocks:
            response += "\n```"
    if "Code Example" in response and not re.search(r'```', response):
        response += "\n**Code Example (Python):**\n```python\nprint(\"Hello, world!\")  # Default example\n```"
    if not re.search(r"\*\*Code Example $$   Python   $$:\*\*", response):
        response = re.sub(r"```python", "**Code Example (Python):**\n```python", response)
    if not re.search(r"\*\*Code Example $$   Java   $$:\*\*", response):
        response = re.sub(r"```java", "**Code Example (Java):**\n```java", response)
    if not re.search(r"\*\*Code Example $$   C\+\+   $$:\*\*", response):
        response = re.sub(r"```cpp", "**Code Example (C++):**\n```cpp", response)
    if not re.search(r"\*\*Code Example $$   JavaScript   $$:\*\*", response):
        response = re.sub(r"```javascript", "**Code Example (JavaScript):**\n```javascript", response)
    if not re.search(r"\*\*Code Example $$   TypeScript   $$:\*\*", response):
        response = re.sub(r"```typescript", "**Code Example (TypeScript):**\n```typescript", response)
    if not re.search(r"\*\*Code Example $$   Go   $$:\*\*", response):
        response = re.sub(r"```go", "**Code Example (Go):**\n```go", response)
    if not re.search(r"\*\*Code Example $$   Rust   $$:\*\*", response):
        response = re.sub(r"```rust", "**Code Example (Rust):**\n```rust", response)
    response = re.sub(r"```", "\n```", response)
    if classify_query(response.split("\n")[0]) == "greeting":
        response = re.split(r'(?<=[.!?])\s+', response)[0] + "."
    if len(response) > 3000:
        response = response[:3000] + "... (response trimmed)."
    elif len(response) < 50:
        response += " Please provide more details if needed."
    return response

def query_mistral(chat_id, prompt, deep_dive=False):
    mode = classify_query(prompt)
    last_response = get_previous_response(chat_id) if deep_dive else None
    chat_history = get_chat_history(chat_id)[-MAX_HISTORY:]
    SYSTEM_PROMPT = (
        "You are AlgoAI, an expert in coding, algorithms, and system design. Your responses must always be:\n"
        "- **Fully detailed and well-structured** with clear, numbered sections as instructed.\n"
        "- **Professional & technical**, assuming the user is a developer or engineer.\n"
        "- **Highly interactive** with logical suggestions, improvements, and follow-up questions.\n"
        "- **Context-aware**, referencing previous messages to maintain a natural flow.\n"
        "- **Focused on best practices, optimization, and scalability considerations.**\n\n"
        "**Strict Response Format (for all modes):**\n"
        "- Use **bold (**text**) for key terms and section headers.\n"
        "- End each response with a specific follow-up question as instructed.\n"
        "- Do not repeat previous responses unless explicitly building on them.\n\n"
        "**Mode-Specific Rules:**\n"
        "- For tech queries, include all sections: Concept Explanation, Complexity Analysis, Example Implementation, Alternative Approaches, Performance Optimization, Next Steps.\n"
        "- For general queries or deep dives, adapt the structure but maintain depth and interactivity.\n"
        "- If a code example is requested but not provided, include a default example (e.g., a simple Python `print()` statement).\n"
    )
    full_prompt = f"<s>[INST] {SYSTEM_PROMPT}\n\n### Conversation Context:\n"
    if chat_history:
        full_prompt += "Here’s our recent chat for context:\n"
        for msg in chat_history:
            full_prompt += f"User: {msg['user']}\nAlgoAI: {msg['ai']}\n"
    else:
        full_prompt += "This is a fresh start—no chat history yet.\n"
    full_prompt += f"\n### User Prompt:\nUser: {prompt}\n\n### Instructions:\n"
    welcome_shown = has_welcome_been_shown(chat_id)
    if mode == "greeting" and welcome_shown:
        return "Please provide your next query."
    if mode == "greeting":
        full_prompt += (
            "Respond with a concise, professional greeting (1 sentence). "
            "End with: **Please provide your next query.** [/INST]"
        )
        max_tokens = 100
        temp = 0.3
    elif mode == "tech":
        language_match = re.search(r'\b(in|using)\s*(python|java|c\+\+|javascript|typescript|go|rust|ruby|php|kotlin|swift)\b', prompt, re.IGNORECASE)
        preferred_language = language_match.group(2).lower() if language_match else "python"
        if deep_dive and last_response:
            full_prompt += (
                f"Build on the previous response: **Previous response: {last_response}**\n"
                "**1. Detailed Analysis**: Analyze the last response in depth (2-3 sentences).\n"
                "**2. Improvements**: Suggest specific optimizations or debugging tips if code was included (2-3 sentences).\n"
                "**3. Alternatives**: Compare with at least one alternative approach (1-2 sentences).\n"
                "**4. Next Steps**: Propose a related topic or question (1 sentence).\n"
                "No repeats—keep it new. End with: **Would you like a different approach or more depth?** [/INST]"
            )
            max_tokens = 1200
            temp = 0.2
        else:
            full_prompt += (
                "Provide MIXTRAL clear, step-by-step technical response:\n"
                "**1. Clarify Intent**: State the intent and preferred language (**{preferred_language}** if specified, else Python) (1 sentence).\n"
                "**2. Concept Explanation**: Explain the algorithmic foundation with theory (2-3 sentences).\n"
                "**3. Complexity Analysis**: Break down the logical structure with Big-O time and space complexity (2-3 sentences).\n"
                "**4. Example Implementation**: Show a clean, well-commented code example in {preferred_language} with ``` blocks (2-3 sentences).\n"
                "**5. Alternative Approaches**: Suggest at least one alternative solution (1-2 sentences).\n"
                "**6. Performance Optimization**: Offer ways to improve efficiency or scalability (1-2 sentences).\n"
                "**7. Next Steps**: Encourage exploration of related topics (1 sentence).\n"
                "End with: **Would you like a deeper explanation?** [/INST]"
            )
            max_tokens = 1000
            temp = 0.2
    else:
        if deep_dive and last_response:
            full_prompt += (
                f"Expand on the previous response: **Previous response: {last_response}**\n"
                "**1. In-Depth Analysis**: Provide detailed insights or examples (2-3 sentences).\n"
                "**2. Improvements**: Suggest enhancements or related considerations (1-2 sentences).\n"
                "**3. Next Steps**: Encourage further exploration (1 sentence).\n"
                "No repeats—keep it new. End with: **Would you like to dig deeper?** [/INST]"
            )
            max_tokens = 600
        else:
            if "joke" in prompt.lower():
                full_prompt += (
                    "Provide a lighthearted joke, followed by a brief response related to the query (2-3 sentences).\n"
                    "End with: **Do you have further questions or would you like a deeper explanation on a topic?** [/INST]"
                )
            else:
                full_prompt += (
                    "Give a clear, concise response:\n"
                    "**1. Direct Answer**: Answer the question directly (1-2 sentences).\n"
                    "**2. Context**: Add relevant context if applicable (1 sentence).\n"
                    "End with: **Do you have further questions or would you like a deeper explanation on a topic?** [/INST]"
                )
            max_tokens = 400
        temp = 0.3
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"} if API_KEY else {}
    data = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temp,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    }
    for attempt in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=data, timeout=30, verify=certifi.where())
            response.raise_for_status()
            json_response = response.json()
            bot_response = None
            if isinstance(json_response, list) and len(json_response) > 0 and "generated_text" in json_response[0]:
                bot_response = json_response[0]["generated_text"].strip()
            elif "generated_text" in json_response:
                bot_response = json_response["generated_text"].strip()
            else:
                return "Apologies, I couldn’t process that due to an API glitch. Please try again."
            if bot_response:
                if "[/INST]" in bot_response:
                    bot_response = bot_response.split("[/INST]", 1)[1].strip()
                bot_response = bot_response.replace(SYSTEM_PROMPT, '').strip()
                if not bot_response:
                    return "No response generated. Please provide more details and try again."
                return bot_response
            return "An error occurred. Please retry your query."
        except requests.RequestException as e:
            print(f"API error, attempt {attempt + 1}: {e}")
            if attempt == 2:
                return f"Error: API failed—{str(e)}. Please check your API key or model access."
            time.sleep(2 ** attempt)

# Routes
@app.route("/")
def home():
    return jsonify({"message": "Welcome to AlgoAI! Use /query to begin."}), 200

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/query", methods=["POST"])
def get_response():
    try:
        data = request.get_json()
        user_query = data.get("query", "No query provided.")
        chat_id = data.get("chat_id", str(uuid.uuid4()))
        deep_dive = data.get("deep_dive", False)
        mistral_response = query_mistral(chat_id, user_query, deep_dive)
        formatted_response = format_response(mistral_response)
        store_chat(chat_id, user_query, formatted_response)
        return jsonify({"response": formatted_response, "chat_id": chat_id})
    except Exception as e:
        print(f"Error in get_response: {e}")
        return jsonify({"error": f"API request failed—{str(e)}", "chat_id": chat_id}), 500

@app.route("/new_chat", methods=["POST"])
def new_chat():
    try:
        chat_id = str(uuid.uuid4())
        welcome_messages = [
            "Welcome to AlgoAI! 🚀 Ready to explore algorithms, coding, or problem-solving? What would you like to discuss?",
            "Greetings! I’m AlgoAI, your intelligent assistant for technical insights. What topic can I assist you with today?",
            "Hello! You’ve activated AlgoAI—let’s dive into the world of AI and code. What’s your first question?",
            "Welcome back! AlgoAI is here to help with your next challenge. What would you like to explore?"
        ]
        existing_chats = get_all_chats()
        is_returning = len(existing_chats) > 0
        greeting = random.choice(welcome_messages)
        if is_returning:
            greeting = "Welcome back! AlgoAI is ready to assist with your next challenge. What would you like to explore?"
        store_chat(chat_id, "", greeting, title="New Chat", welcome_shown=1)
        return jsonify({"chat_id": chat_id, "greeting": greeting})
    except Exception as e:
        print(f"Error in new_chat: {e}")
        return jsonify({"error": f"Failed to create new chat—{e}"}), 500

@app.route("/reset_chat", methods=["POST"])
def reset_chat():
    session = Session()
    try:
        data = request.get_json()
        chat_id = data.get("chat_id")
        if not chat_id:
            return jsonify({"error": "No chat_id provided."}), 400
        session.query(ChatHistory).filter_by(chat_id=chat_id).delete()
        session.commit()
        return jsonify({"message": f"Chat {chat_id} has been reset.", "chat_id": chat_id})
    except Exception as e:
        print(f"Error in reset_chat: {e}")
        session.rollback()
        return jsonify({"error": f"Reset failed—{e}"}), 500
    finally:
        session.close()

@app.route("/get_current_chat", methods=["GET"])
def get_current_chat():
    try:
        chat_id = request.args.get("chat_id", str(uuid.uuid4()))
        history = get_chat_history(chat_id)
        title = get_chat_title(chat_id) or f"Chat #{chat_id}"
        return jsonify({"chat_id": chat_id, "title": title, "history": history})
    except Exception as e:
        print(f"Error in get_current_chat: {e}")
        return jsonify({"error": f"Failed to fetch chat—{e}"}), 500

@app.route("/get_chat_history", methods=["GET"])
def get_chat_history_endpoint():
    try:
        chats = get_all_chats()
        return jsonify({"chats": chats})
    except Exception as e:
        print(f"Error in get_chat_history_endpoint: {e}")
        return jsonify({"error": f"Failed to fetch chat history—{e}"}), 500

@app.route("/get_chat/<identifier>", methods=["GET"])
def get_chat(identifier):
    try:
        chat_data = get_chat_by_title_or_id(identifier)
        if chat_data:
            return jsonify(chat_data)
        return jsonify({"error": "Chat not found!", "chat_id": identifier}), 404
    except Exception as e:
        print(f"Error in get_chat: {e}")
        return jsonify({"error": f"Chat fetch failed—{e}", "chat_id": identifier}), 500

@app.route("/update_chat/<chat_id>", methods=["POST"])
def update_chat(chat_id):
    try:
        if not chat_id or not isinstance(chat_id, str):
            return jsonify({"error": "Invalid chat_id!", "chat_id": None}), 400
        data = request.get_json()
        user_msg = data.get("user_msg", "")
        ai_msg = data.get("ai_msg", "")
        title = data.get("title")
        store_chat(chat_id, user_msg, ai_msg, title)
        return jsonify({"message": f"Chat {chat_id} updated successfully!", "chat_id": chat_id, "title": title})
    except Exception as e:
        print(f"Error in update_chat: {e}")
        return jsonify({"error": f"Update failed—{e}", "chat_id": chat_id}), 500

@app.route('/update_chat_title', methods=['POST'])
def update_chat_title():
    data = request.get_json()
    chat_id = data.get('chat_id')
    new_title = data.get('title')

    if not chat_id or not new_title:
        return jsonify({"error": "chat_id and title are required"}), 400

    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("UPDATE chats SET title = ? WHERE chat_id = ?", (new_title, chat_id))
    conn.commit()
    conn.close()

    return jsonify({"message": "Chat title updated successfully"})
    
@app.route("/test")
def test():
    return jsonify({"message": "Backend is operational!"})