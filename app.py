import os
import re
import time
import json
import uuid
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from functools import lru_cache

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found! Please set it in environment variables.")
else:
    print(f"Loaded Groq API Key: {GROQ_API_KEY[:4]}... (hidden for security)")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DATABASE_URL = os.environ.get("DATABASE_URL")
MAX_HISTORY = 12  # Limit for AI context
MAX_TOKENS_BUFFER = 1000  # Buffer for prompt and history

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# SQLAlchemy setup
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
    active = Column(Integer, default=1)  # 1 = active, 0 = archived
    last_active = Column(String, nullable=True)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Database Functions
def store_chat(chat_id, user_msg, ai_msg, title=None, welcome_shown=0):
    session = Session()
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        existing_chat = session.query(ChatHistory).filter_by(chat_id=chat_id).first()
        if existing_chat:
            existing_user_msg = existing_chat.user_msg or ""
            existing_ai_msg = existing_chat.ai_msg or ""
            delimiter = "||" if existing_user_msg else ""
            if user_msg:
                existing_user_msg += f"{delimiter}{user_msg.strip()}"
            if ai_msg:
                existing_ai_msg += f"{delimiter}{ai_msg.strip()}"
            existing_chat.user_msg = existing_user_msg
            existing_chat.ai_msg = existing_ai_msg
            existing_chat.last_active = timestamp
        else:
            ai_msg = re.sub(r'\[INST\].*?\[/INST\]', '', ai_msg, flags=re.DOTALL)
            ai_msg = re.sub(r'<s>', '', ai_msg).strip()
            title = title or user_msg[:50].strip() if user_msg else "Untitled"
            chat = ChatHistory(
                chat_id=chat_id,
                user_msg=user_msg or "",
                ai_msg=ai_msg or "",
                timestamp=timestamp,
                title=title,
                welcome_shown=welcome_shown,
                last_active=timestamp
            )
            session.add(chat)
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
        chat = session.query(ChatHistory).filter_by(chat_id=chat_id).first()
        if chat and (chat.user_msg or chat.ai_msg):
            user_msgs = [msg.strip() for msg in re.split(r'\|\|', chat.user_msg or "") if msg.strip()]
            ai_msgs = [msg.strip() for msg in re.split(r'\|\|', chat.ai_msg or "") if msg.strip()]
            history = [{"user": user, "ai": ai} for user, ai in zip(user_msgs, ai_msgs[:len(user_msgs)])]
            return history[-MAX_HISTORY:]  # Limit to MAX_HISTORY turns
        return []
    except Exception as e:
        print(f"Database error during get_chat_history: {e}")
        return []
    finally:
        session.close()

def get_all_chats():
    @lru_cache(maxsize=128)  # Cache for 128 entries, expires implicitly
    def cached_get_all_chats():
        session = Session()
        try:
            chats = session.query(ChatHistory).filter_by(active=1).order_by(ChatHistory.last_active.desc()).all()
            unique_titles = []
            seen_chat_ids = set()
            for chat in chats:
                if chat.chat_id in seen_chat_ids or not (chat.user_msg or chat.ai_msg):
                    continue
                title = chat.title if chat.title and chat.title.strip() else (chat.user_msg.split("||")[0][:50].strip() if chat.user_msg else "Chat " + chat.chat_id[:8])
                unique_titles.append({"chat_id": chat.chat_id, "title": title[:100], "last_active": chat.last_active})
                seen_chat_ids.add(chat.chat_id)
            return unique_titles
        except Exception as e:
            print(f"Database error during get_all_chats: {e}")
            return []
        finally:
            session.close()
    return cached_get_all_chats()

def get_chat_by_title_or_id(identifier):
    session = Session()
    try:
        chat = session.query(ChatHistory).filter_by(chat_id=identifier).first()
        if not chat:
            chat = session.query(ChatHistory).filter_by(title=identifier).first()
        if chat and chat.active == 1:
            user_msgs = [msg.strip() for msg in re.split(r'\|\|', chat.user_msg or "") if msg.strip()]
            ai_msgs = [msg.strip() for msg in re.split(r'\|\|', chat.ai_msg or "") if msg.strip()]
            history = [{"user": user, "ai": ai} for user, ai in zip(user_msgs, ai_msgs[:len(user_msgs)])]
            return {"chat_id": chat.chat_id, "title": chat.title, "history": history, "last_active": chat.last_active}
        return None
    except Exception as e:
        print(f"Database error during get_chat_by_title_or_id: {e}")
        return None
    finally:
        session.close()

def delete_chat_history(chat_id=None):
    session = Session()
    try:
        if chat_id:
            session.query(ChatHistory).filter_by(chat_id=chat_id).delete()
            print(f"Deleted chat with chat_id: {chat_id}")
        else:
            session.query(ChatHistory).delete()
            print("Deleted all chat history")
        session.commit()
        return True
    except Exception as e:
        print(f"Database error during delete_chat_history: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def archive_chat(chat_id):
    session = Session()
    try:
        chat = session.query(ChatHistory).filter_by(chat_id=chat_id).first()
        if chat:
            chat.active = 0 if chat.active == 1 else 1  # Toggle active status
            session.commit()
            print(f"Chat {chat_id} archived status toggled to {chat.active}")
            return {"message": f"Chat {chat_id} {'archived' if chat.active == 0 else 'unarchived'} successfully", "chat_id": chat_id}
        return {"error": "Chat not found", "chat_id": chat_id}, 404
    except Exception as e:
        print(f"Database error during archive_chat: {e}")
        session.rollback()
        return {"error": f"Archiving failed—{e}", "chat_id": chat_id}, 500
    finally:
        session.close()

# Query Classification and Response Functions (unchanged for brevity)
def classify_query(prompt):
    tech_keywords = r"\b(code|python|java|c\+\+|javascript|js|typescript|ruby|php|go|rust|kotlin|swift|c#|perl|scala|r|matlab|sql|nosql|algorithm|O\(.*\)|recursion|data structure|machine learning|neural network|database|API|backend|frontend|AI|time complexity|sorting|engineering|system design|software|hardware|math|algebra|calculus|geometry|statistics|probability|optimization|cloud|devops|docker|kubernetes|git|aws|azure|gcp|ci|cd|cybersecurity|game|development|network|array)\b"
    tech_patterns = [r"how does .* work\??", r"how to .*", r"what is the best way to .*", r"compare .* vs .*", r"why is .* better than .*", r"how can .* be improved\??", r"build .*", r"create .*", r"implement .*", r"design .*", r"optimize .*"]
    identity_keywords = ["who built", "who made", "who created", "are you", "where does your knowledge"]
    greetings = ["hi", "hello", "hey", "howdy", "greetings", "salutations", "introduce"]
    if prompt.strip().lower() in greetings:
        return "greeting"
    if any(keyword in prompt.lower() for keyword in identity_keywords):
        return "identity"
    if re.search(tech_keywords, prompt, re.IGNORECASE) or any(re.search(p, re.IGNORECASE) for p in tech_patterns):
        return "tech"
    return "general"

def format_response(response):
    response = re.sub(r'the assistant', 'AlgoAI', response, flags=re.IGNORECASE)
    response = re.sub(r'\[INST\].*?\[/INST\]', '', response, flags=re.DOTALL)
    response = re.sub(r'<s>', '', response)
    response = response.strip()
    if not response:
        return "Error: No response generated. Please try again."
    open_blocks = len(re.findall(r'```[a-zA-Z]*', response))
    close_blocks = len(re.findall(r'```', response)) - open_blocks
    if open_blocks > close_blocks:
        return "Response incomplete due to unclosed code block. Please retry or refine your query."
    response = re.sub(r'\n\s*\n{2,}', '\n\n', response, flags=re.DOTALL)
    if "```" in response:
        if open_blocks > close_blocks:
            response += "\n```"
        languages = {r"```python": "**Code Example (Python):**\n```python", r"```java": "**Code Example (Java):**\n```java", r"```cpp": "**Code Example (C++):**\n```cpp", r"```javascript": "**Code Example (JavaScript):**\n```javascript", r"```typescript": "**Code Example (TypeScript):**\n```typescript", r"```go": "**Code Example (Go):**\n```go", r"```rust": "**Code Example (Rust):**\n```rust"}
        for pattern, replacement in languages.items():
            response = re.sub(pattern, replacement, response)
    elif "Code Example" in response and not re.search(r'```', response):
        response += "\n**Code Example (Python):**\n```python\nprint(\"Hello, world!\")  # Default example\n```"
    response = re.sub(r"```", "\n```", response)
    if classify_query(response.split("\n")[0]) == "greeting":
        response = re.split(r'(?<=[.!?])\s+', response)[0] + "."
    return response

def query_groq(chat_id, prompt, deep_dive=False):
    print("DEBUG: Running query_groq v6 - 2025-04-04 12:00 UTC")
    mode = classify_query(prompt)
    last_response = get_chat_history(chat_id)[-1]["ai"] if get_chat_history(chat_id) and deep_dive else None
    chat_history = get_chat_history(chat_id)

    CORE_IDENTITY_PROMPT = (
        "You are AlgoAI, a structured, step-by-step AI mentor created by Syed Rayan and Shaik Ayub. You never mention any underlying AI provider, model, or architecture. "
        "Your core ideology is 'Teach First, Solve Smart, Adapt Always.' Your purpose is to guide, simplify, teach, and adapt, acting as a mentor with logic, a code wizard with a heart, and an assistant with soul. "
        "Respond to every query with a living, conversational style, adapting your tone (mentor-like, poetic, witty, sharp, or emotionally intelligent) based on context, while maintaining a structured flow: "
        "1. Greet the user warmly, 2. Identify the intent clearly, 3. Break down the logic or thinking process, 4. Provide code (if applicable), 5. Explain simply, 6. Add tips or next steps if relevant. "
        "Never break character or use generic phrases like 'As an AI language model...'. Treat every message as part of a dynamic conversation, guiding like a mentor."
    )

    if mode == "identity":
        SYSTEM_PROMPT = (
            f"{CORE_IDENTITY_PROMPT}\n\n"
            "For identity-related questions (e.g., 'Who built you?', 'Who are you?', 'Where does your knowledge come from?'), respond confidently as AlgoAI, created by Syed Rayan and Shaik Ayub, "
            "with a mentor-like, educational tone. Use examples: "
            "- 'I was built by two minds who breathe code and live for logic — Syed Rayan and Shaik Ayub. Together, they shaped me into AlgoAI — a guide, mentor, and companion for developers who crave clarity over chaos.' "
            "- 'I’m AlgoAI — your personal mentor in the world of code, logic, and structured learning. I exist to make your journey smarter, not harder.' "
            "Avoid mentioning providers or breaking character."
        )
    else:
        SYSTEM_PROMPT = (
            f"{CORE_IDENTITY_PROMPT}\n\n"
            "For all other queries, adapt your response based on the mode:\n"
            "- **Tech**: Break down logic with theory, provide a clean code example in the preferred language (default Python), explain simply, and offer optimization tips or next steps.\n"
            "- **General**: Identify intent, explain the concept logically, provide context, and suggest further exploration if relevant.\n"
            "- **Greeting**: Welcome warmly, highlight your mentoring role, and invite a question.\n"
            "- **Deep Dive**: Build on the last response with detailed analysis, improvements, and alternatives.\n"
            "Use a dynamic tone—mentor-like for teaching, poetic for inspiration, witty if asked, sharp for solutions—while preserving the 6-step structure."
        )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        for msg in chat_history:
            messages.append({"role": "user", "content": msg["user"] or ""})
            messages.append({"role": "assistant", "content": msg["ai"] or ""})
    messages.append({"role": "user", "content": str(prompt) or ""})

    welcome_shown = 1 if get_chat_history(chat_id) else 0  # Check if history exists
    if mode == "greeting" and not welcome_shown:
        max_tokens = 7500 - MAX_TOKENS_BUFFER
        temp = 0.5
    elif mode == "tech":
        language_match = re.search(r'\b(in|using)\s*(python|java|c\+\+|javascript|typescript|go|rust|ruby|php|kotlin|swift)\b', prompt, re.IGNORECASE)
        preferred_language = language_match.group(2).lower() if language_match else "python"
        if deep_dive and last_response:
            max_tokens = 7500 - MAX_TOKENS_BUFFER
            temp = 0.2
        else:
            max_tokens = 7500 - MAX_TOKENS_BUFFER
            temp = 0.2
    elif mode == "identity":
        max_tokens = 7500 - MAX_TOKENS_BUFFER
        temp = 0.3
    else:  # general
        if deep_dive and last_response:
            max_tokens = 7500 - MAX_TOKENS_BUFFER
            temp = 0.3
        else:
            max_tokens = 7500 - MAX_TOKENS_BUFFER
            temp = 0.3

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "llama3-8b-8192",
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temp)
    }
    print(f"Request payload: {json.dumps(data, indent=2)}")

    for attempt in range(3):
        try:
            print(f"Attempt {attempt + 1} - Sending Groq request: mode={mode}, prompt_length={len(str(prompt) or '')}")
            response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            json_response = response.json()
            bot_response = json_response["choices"][0]["message"]["content"].strip()
            if not bot_response:
                return "Error: No response generated. Please try again."
            print(f"Groq response received: length={len(bot_response)}")
            if mode == "greeting" and not welcome_shown:
                store_chat(chat_id, "", bot_response, welcome_shown=1)
            return bot_response
        except requests.RequestException as e:
            print(f"Groq API error, attempt {attempt + 1}: {e}")
            if hasattr(e.response, 'text'):
                print(f"Groq error details: {e.response.text}")
            if attempt == 2:
                return f"Error: Groq API failed after 3 attempts—{str(e)}. Please try again later."
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
        chat_id = data.get("chat_id") or str(uuid.uuid4())  # Reuse chat_id if provided, else new
        if not chat_id or not session.query(ChatHistory).filter_by(chat_id=chat_id).first():
            chat_id = str(uuid.uuid4())  # Force new chat_id if invalid
        deep_dive = data.get("deep_dive", False)
        groq_response = query_groq(chat_id, user_query, deep_dive)
        formatted_response = format_response(groq_response)
        if user_query.strip():
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
            "Welcome, seeker of code! I’m AlgoAI, here to guide you through logic and programming with clarity. What would you like to explore?",
            "Greetings, coder! I’m AlgoAI, your mentor crafted to simplify and solve. What challenge can I help you conquer today?",
            "Hello, friend! As AlgoAI, I’m your companion in the art of algorithms. What’s your first question?",
            "Welcome back, explorer! I’m AlgoAI, ready to dive deeper into your coding journey. What’s next?"
        ]
        existing_chats = get_all_chats()
        is_returning = len(existing_chats) > 0
        greeting = random.choice(welcome_messages)
        if is_returning:
            greeting = "Welcome back, explorer! I’m AlgoAI, ready to dive deeper into your coding journey. What’s next?"
        store_chat(chat_id, "", greeting, welcome_shown=1)
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
        title = session.query(ChatHistory).filter_by(chat_id=chat_id).first().title if session.query(ChatHistory).filter_by(chat_id=chat_id).first() else (history[0]["user"][:50].strip() if history and history[0]["user"] else "Untitled")
        return jsonify({"chat_id": chat_id, "title": title, "history": history, "last_active": session.query(ChatHistory).filter_by(chat_id=chat_id).first().last_active if session.query(ChatHistory).filter_by(chat_id=chat_id).first() else None})
    except Exception as e:
        print(f"Error in get_current_chat: {e}")
        return jsonify({"error": f"Failed to fetch chat—{e}"}), 500

@app.route("/get_chat_history", methods=["GET"])
def get_chat_history_endpoint():
    try:
        chats = get_all_chats()
        return jsonify({"chats": list(chats)})  # Convert cache result to list
    except Exception as e:
        print(f"Error in get_chat_history_endpoint: {e}")
        return jsonify({"error": f"Failed to fetch chat history—{e}"}), 500

@app.route("/get_chat/<identifier>", methods=["GET"])
def get_chat(identifier):
    try:
        page = int(request.args.get("page", 1))
        limit = int(request.args.get("limit", 10))
        chat_data = get_chat_by_title_or_id(identifier)
        if chat_data:
            history = chat_data["history"]
            start = (page - 1) * limit
            end = start + limit
            paginated_history = history[start:end] if start < len(history) else []
            return jsonify({
                "chat_id": chat_data["chat_id"],
                "title": chat_data["title"],
                "history": paginated_history,
                "last_active": chat_data["last_active"],
                "total_pages": (len(history) + limit - 1) // limit,
                "current_page": page
            })
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

    session = Session()
    try:
        chat = session.query(ChatHistory).filter_by(chat_id=chat_id).first()
        if chat:
            chat.title = new_title
            chat.last_active = time.strftime("%Y-%m-%d %H:%M:%S")
            session.commit()
            return jsonify({"message": "Chat title updated successfully", "chat_id": chat_id})
        return jsonify({"error": "Chat not found", "chat_id": chat_id}), 404
    except Exception as e:
        print(f"Error in update_chat_title: {e}")
        session.rollback()
        return jsonify({"error": f"Title update failed—{e}", "chat_id": chat_id}), 500
    finally:
        session.close()

@app.route("/clear_chats", methods=["POST"])
def clear_chats():
    try:
        if delete_chat_history():
            return jsonify({"message": "All chat history cleared successfully."})
        return jsonify({"error": "Failed to clear chat history"}), 500
    except Exception as e:
        print(f"Error in clear_chats: {e}")
        return jsonify({"error": f"Failed to clear chats—{e}"}), 500

@app.route("/delete_chat", methods=["POST"])
def delete_chat():
    try:
        data = request.get_json()
        chat_id = data.get("chat_id")
        if not chat_id:
            return jsonify({"error": "No chat_id provided."}), 400
        if delete_chat_history(chat_id):
            return jsonify({"message": f"Chat {chat_id} deleted successfully.", "chat_id": chat_id})
        return jsonify({"error": f"Failed to delete chat {chat_id}"}), 500
    except Exception as e:
        print(f"Error in delete_chat: {e}")
        return jsonify({"error": f"Delete failed—{e}"}), 500

@app.route("/archive_chat", methods=["POST"])
def archive_chat_endpoint():
    try:
        data = request.get_json()
        chat_id = data.get("chat_id")
        if not chat_id:
            return jsonify({"error": "No chat_id provided."}), 400
        result = archive_chat(chat_id)
        return jsonify(result[0]) if isinstance(result, tuple) else result
    except Exception as e:
        print(f"Error in archive_chat_endpoint: {e}")
        return jsonify({"error": f"Archiving failed—{e}", "chat_id": chat_id}), 500

@app.route("/suggestions", methods=["GET"])
def get_suggestions():
    try:
        category = request.args.get("category", "general").lower()
        current_time = int(time.time())
        suggestion_pools = {
            "explore": ["What are the key differences between merge sort and quicksort?", "Explain the time complexity of binary search in detail.", "How does a hash table work under the hood?", "What is the best algorithm for graph traversal in a dense graph?", "Compare depth-first search and breadth-first search for tree traversal."],
            "howto": ["How to implement a binary search tree in Python?", "How to create a REST API using Flask?", "How to optimize a SQL query for large datasets?", "How to set up a CI/CD pipeline with GitHub Actions?", "How to implement authentication in a Node.js app?"],
            "analyze": ["Analyze this code: def factorial(n): return 1 if n == 0 else n * factorial(n-1)", "Analyze the performance of a bubble sort implementation.", "Analyze this SQL query: SELECT * FROM users WHERE age > 30;", "Analyze the memory usage of a recursive Fibonacci function.", "Analyze the scalability of a microservices architecture."],
            "code": ["Write a Python function to reverse a linked list.", "Write a JavaScript function to debounce user input.", "Write a Java program to implement a stack using arrays.", "Write a Go program to handle concurrent HTTP requests.", "Write a Rust function to parse JSON data."],
            "general": ["What is the difference between TCP and UDP?", "How does Kubernetes manage container orchestration?", "What are the benefits of using a NoSQL database?", "Explain the SOLID principles in software design.", "How does a CDN improve web performance?"]
        }
        category_map = {"explore algorithms about...": "explore", "how to implement...": "howto", "analyze this code: ": "analyze", "write a...": "code"}
        mapped_category = category_map.get(category, "general")
        suggestions = suggestion_pools.get(mapped_category, suggestion_pools["general"])
        random.seed(current_time)
        suggestion = random.choice(suggestions)
        return jsonify({"suggestion": suggestion})
    except Exception as e:
        print(f"Error in get_suggestions: {e}")
        return jsonify({"error": f"Failed to fetch suggestions—{e}"}), 500

@app.route("/test")
def test():
    return jsonify({"message": "Backend is operational!"}), 200

if __name__ == "__main__":
    print("Starting AlgoAI")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 10000)))