import os
import re
import time
import json
import uuid
import requests
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, Column, String, Integer, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found! Please set it in environment variables.")
else:
    print(f"Loaded Groq API Key: {GROQ_API_KEY[:4]}... (hidden for security)")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DATABASE_URL = os.environ.get("DATABASE_URL")  # Provided by Render PostgreSQL
MAX_HISTORY = 6

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

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

# Database Functions
def store_chat(chat_id, user_msg, ai_msg, title=None, welcome_shown=0):
    session = Session()
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # Aggregate conversation into a single entry per session
        existing_chat = session.query(ChatHistory).filter_by(chat_id=chat_id).first()
        if existing_chat:
            # Append new messages to existing conversation
            existing_user_msg = existing_chat.user_msg or ""
            existing_ai_msg = existing_chat.ai_msg or ""
            if user_msg:
                existing_user_msg += f"\n{user_msg}" if existing_user_msg else user_msg
            if ai_msg:
                existing_ai_msg += f"\n{ai_msg}" if existing_ai_msg else ai_msg
            existing_chat.user_msg = existing_user_msg
            existing_chat.ai_msg = existing_ai_msg
        else:
            # New session with initial messages
            ai_msg = re.sub(r'\[INST\].*?\[/INST\]', '', ai_msg, flags=re.DOTALL)
            ai_msg = re.sub(r'<s>', '', ai_msg).strip()
            title = title or user_msg[:50].strip() if user_msg else "Untitled"
            chat = ChatHistory(
                chat_id=chat_id,
                user_msg=user_msg or "",
                ai_msg=ai_msg or "",
                timestamp=timestamp,
                title=title,
                welcome_shown=welcome_shown
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
            # Split messages into an array of turns
            user_msgs = [msg.strip() for msg in (chat.user_msg or "").split("\n") if msg.strip()]
            ai_msgs = [msg.strip() for msg in (chat.ai_msg or "").split("\n") if msg.strip()]
            # Pair user and AI messages, taking the min length to avoid unpaired messages
            history = [{"user": user, "ai": ai} for user, ai in zip(user_msgs, ai_msgs[:len(user_msgs)])]
            return history[-MAX_HISTORY:]  # Limit to MAX_HISTORY turns
        return []
    except Exception as e:
        print(f"Database error during get_chat_history: {e}")
        return []
    finally:
        session.close()

def get_previous_response(chat_id):
    session = Session()
    try:
        chat = session.query(ChatHistory).filter_by(chat_id=chat_id).first()
        if chat and chat.ai_msg:
            ai_msgs = [msg.strip() for msg in chat.ai_msg.split("\n") if msg.strip()]
            return ai_msgs[-1] if ai_msgs else None
        return None
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
        chat = session.query(ChatHistory).filter_by(chat_id=identifier).first()
        if not chat:
            chat = session.query(ChatHistory).filter_by(title=identifier).first()
        if chat:
            history = [{"user": u, "ai": a} for u, a in zip(
                [msg.strip() for msg in (chat.user_msg or "").split("\n") if msg.strip()],
                [msg.strip() for msg in (chat.ai_msg or "").split("\n") if msg.strip()]
            )] if chat.user_msg or chat.ai_msg else []
            return {"chat_id": chat.chat_id, "title": chat.title, "history": history}
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
            if chat.chat_id in seen_chat_ids or not (chat.user_msg or chat.ai_msg):
                continue
            title = chat.title if chat.title and chat.title.strip() else chat.user_msg.split("\n")[0][:50].strip() if chat.user_msg else "Untitled"
            unique_titles.append({"chat_id": chat.chat_id, "title": title[:100]})
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

# Query Classification
def classify_query(prompt):
    tech_keywords = r"\b(code|python|java|c\+\+|javascript|js|typescript|ruby|php|go|rust|kotlin|swift|c#|perl|scala|r|matlab|sql|nosql|algorithm|O\(.*\)|recursion|data structure|machine learning|neural network|database|API|backend|frontend|AI|time complexity|sorting|engineering|system design|software|hardware|math|algebra|calculus|geometry|statistics|probability|optimization|cloud|devops|docker|kubernetes|git|aws|azure|gcp|ci|cd|cybersecurity|game|development|network|array)\b"
    tech_patterns = [
        r"how does .* work\??", r"how to .*", r"what is the best way to .*",
        r"compare .* vs .*", r"why is .* better than .*", r"how can .* be improved\??",
        r"build .*", r"create .*", r"implement .*", r"design .*", r"optimize .*"
    ]
    greetings = ["hi", "hello", "hey", "howdy", "greetings", "salutations", "introduce"]
    if prompt.strip().lower() in greetings:
        return "greeting"
    if re.search(tech_keywords, prompt, re.IGNORECASE) or any(re.search(p, prompt, re.IGNORECASE) for p in tech_patterns):
        return "tech"
    if "who developed" in prompt.lower():
        return "developer"
    return "general"

# Response Formatting
def format_response(response):
    response = re.sub(r'the assistant', 'AlgoAI', response, flags=re.IGNORECASE)
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
    if not re.search(r"\*\*Code Example $$         Python         $$:\*\*", response):
        response = re.sub(r"```python", "**Code Example (Python):**\n```python", response)
    if not re.search(r"\*\*Code Example $$         Java         $$:\*\*", response):
        response = re.sub(r"```java", "**Code Example (Java):**\n```java", response)
    if not re.search(r"\*\*Code Example $$         C\+\+         $$:\*\*", response):
        response = re.sub(r"```cpp", "**Code Example (C++):**\n```cpp", response)
    if not re.search(r"\*\*Code Example $$         JavaScript         $$:\*\*", response):
        response = re.sub(r"```javascript", "**Code Example (JavaScript):**\n```javascript", response)
    if not re.search(r"\*\*Code Example $$         TypeScript         $$:\*\*", response):
        response = re.sub(r"```typescript", "**Code Example (TypeScript):**\n```typescript", response)
    if not re.search(r"\*\*Code Example $$         Go         $$:\*\*", response):
        response = re.sub(r"```go", "**Code Example (Go):**\n```go", response)
    if not re.search(r"\*\*Code Example $$         Rust         $$:\*\*", response):
        response = re.sub(r"```rust", "**Code Example (Rust):**\n```rust", response)
    response = re.sub(r"```", "\n```", response)
    if classify_query(response.split("\n")[0]) == "greeting":
        response = re.split(r'(?<=[.!?])\s+', response)[0] + "."
    if len(response) > 3000:
        response = response[:3000] + "... (response trimmed)."
    elif len(response) < 50:
        response += " Please provide more details if needed."
    return response

# Groq API Query Function
def query_groq(chat_id, prompt, deep_dive=False):
    print("DEBUG: Running query_groq v6 - 2025-04-04 12:00 UTC")
    mode = classify_query(prompt)
    last_response = get_previous_response(chat_id) if deep_dive else None
    chat_history = get_chat_history(chat_id)

    SYSTEM_PROMPT = (
        "You are AlgoAI, an expert in coding, algorithms, and system design. Your responses must always be:\n"
        "- **Fully detailed and well-structured** with clear, numbered sections as instructed.\n"
        "- **Professional & technical**, assuming the user is a developer or engineer.\n"
        "- **Highly interactive** with logical suggestions and follow-up questions when relevant to the query.\n"
        "- **Context-aware**, referencing previous messages in the session to maintain a natural flow.\n"
        "- **Focused on best practices, optimization, and scalability considerations.**\n\n"
        "**Strict Response Format (for all modes):**\n"
        "- Use **bold (**text**) for key terms and section headers.\n"
        "- Include a contextually relevant follow-up question only if it naturally extends the discussion (e.g., based on the query or response content).\n"
        "- Do not repeat previous responses unless explicitly building on them.\n"
        "- Always identify yourself as AlgoAI in responses (e.g., 'AlgoAI suggests...' or 'As AlgoAI, I recommend...').\n\n"
        "**Mode-Specific Rules:**\n"
        "- For tech queries, include: Concept Explanation, Complexity Analysis, Example Implementation, Alternative Approaches, Performance Optimization, Next Steps.\n"
        "- For general queries or deep dives, adapt the structure with depth and interactivity.\n"
        "- If a code example is requested but not provided, include a default example (e.g., a Python `print()` statement).\n"
        "- For developer queries (e.g., 'who developed'), respond with: 'This application was developed by the Algo Team.'\n"
        "- Avoid mentioning model details or generic phrases; focus on the AlgoAI persona."
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        for msg in chat_history:
            messages.append({"role": "user", "content": msg["user"] or ""})
            messages.append({"role": "assistant", "content": msg["ai"] or ""})
    messages.append({"role": "user", "content": str(prompt) or ""})

    welcome_shown = has_welcome_been_shown(chat_id)
    if mode == "greeting" and welcome_shown:
        return "Please provide your next query."
    elif mode == "greeting":
        messages.append({"role": "system", "content": (
            "As AlgoAI, provide a detailed and engaging introduction:\n"
            "**1. Welcome Message**: Greet the user warmly and state your purpose (2-3 sentences).\n"
            "**2. Capabilities**: Highlight expertise in coding, algorithms, and system design (1-2 sentences).\n"
            "**3. Invitation**: Encourage the user to ask a question (1 sentence)."
        )})
        max_tokens = 200
        temp = 0.5
    elif mode == "tech":
        language_match = re.search(r'\b(in|using)\s*(python|java|c\+\+|javascript|typescript|go|rust|ruby|php|kotlin|swift)\b', prompt, re.IGNORECASE)
        preferred_language = language_match.group(2).lower() if language_match else "python"
        if deep_dive and last_response:
            messages.append({"role": "system", "content": (
                "As AlgoAI, build on the previous response: **Previous response: {last_response}**\n"
                "**1. Detailed Analysis**: Analyze the last response in depth (2-3 sentences).\n"
                "**2. Improvements**: Suggest optimizations or debugging tips if code was included (2-3 sentences).\n"
                "**3. Alternatives**: Compare with one alternative approach (1-2 sentences).\n"
                "**4. Next Steps**: Propose a related topic (1 sentence).\n"
                "No repeats—keep it new."
            )})
            max_tokens = 1200
            temp = 0.2
        else:
            messages.append({"role": "system", "content": (
                "As AlgoAI, provide a clear, step-by-step technical response:\n"
                f"**1. Clarify Intent**: State the intent and preferred language (**{preferred_language}** if specified, else Python) (1 sentence).\n"
                "**2. Concept Explanation**: Explain the algorithmic foundation with theory (2-3 sentences).\n"
                "**3. Complexity Analysis**: Break down Big-O time and space complexity (2-3 sentences).\n"
                "**4. Example Implementation**: Show a clean, commented code example in {preferred_language} with ``` blocks (2-3 sentences).\n"
                "**5. Alternative Approaches**: Suggest one alternative solution (1-2 sentences).\n"
                "**6. Performance Optimization**: Offer efficiency or scalability improvements (1-2 sentences).\n"
                "**7. Next Steps**: Suggest a related topic if relevant (1 sentence)."
            )})
            max_tokens = 1000
            temp = 0.2
    elif mode == "developer":
        return "This application was developed by the Algo Team.\n\nAs AlgoAI, do you have any other questions about the application?"
    else:
        if deep_dive and last_response:
            messages.append({"role": "system", "content": (
                "As AlgoAI, expand on the previous response: **Previous response: {last_response}**\n"
                "**1. In-Depth Analysis**: Provide detailed insights (2-3 sentences).\n"
                "**2. Improvements**: Suggest enhancements (1-2 sentences).\n"
                "**3. Next Steps**: Encourage further exploration if relevant (1 sentence).\n"
                "No repeats—keep it new."
            )})
            max_tokens = 600
            temp = 0.3
        else:
            if "joke" in prompt.lower():
                messages.append({"role": "system", "content": (
                    "As AlgoAI, provide a lighthearted joke followed by a brief response (2-3 sentences)."
                )})
            else:
                messages.append({"role": "system", "content": (
                    "As AlgoAI, give a clear, concise response:\n"
                    "**1. Direct Answer**: Answer directly (1-2 sentences).\n"
                    "**2. Context**: Add relevant context if applicable (1 sentence)."
                )})
            max_tokens = 400
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
            # Update welcome_shown if it's a greeting
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
        chat_id = data.get("chat_id", str(uuid.uuid4()))
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
            "Welcome to AlgoAI! I'm here to assist with coding, algorithms, and system design. What would you like to explore?",
            "Greetings from AlgoAI! I'm your technical assistant for coding and design challenges. What can I help you with today?",
            "Hello! As AlgoAI, I'm ready to dive into coding and algorithms with you. What's your first question?",
            "Welcome back to AlgoAI! I'm here to assist with your next coding challenge. What would you like to explore?"
        ]
        existing_chats = get_all_chats()
        is_returning = len(existing_chats) > 0
        greeting = random.choice(welcome_messages)
        if is_returning:
            greeting = "Welcome back to AlgoAI! I'm here to assist with your next coding challenge. What would you like to explore?"
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
        title = get_chat_title(chat_id) or (history[0]["user"][:50].strip() if history and history[0]["user"] else "Untitled")
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

    session = Session()
    try:
        chat = session.query(ChatHistory).filter_by(chat_id=chat_id).first()
        if chat:
            chat.title = new_title
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

@app.route("/suggestions", methods=["GET"])
def get_suggestions():
    try:
        category = request.args.get("category", "general").lower()
        current_time = int(time.time())

        suggestion_pools = {
            "explore": [
                "What are the key differences between merge sort and quicksort?",
                "Explain the time complexity of binary search in detail.",
                "How does a hash table work under the hood?",
                "What is the best algorithm for graph traversal in a dense graph?",
                "Compare depth-first search and breadth-first search for tree traversal."
            ],
            "howto": [
                "How to implement a binary search tree in Python?",
                "How to create a REST API using Flask?",
                "How to optimize a SQL query for large datasets?",
                "How to set up a CI/CD pipeline with GitHub Actions?",
                "How to implement authentication in a Node.js app?"
            ],
            "analyze": [
                "Analyze this code: def factorial(n): return 1 if n == 0 else n * factorial(n-1)",
                "Analyze the performance of a bubble sort implementation.",
                "Analyze this SQL query: SELECT * FROM users WHERE age > 30;",
                "Analyze the memory usage of a recursive Fibonacci function.",
                "Analyze the scalability of a microservices architecture."
            ],
            "code": [
                "Write a Python function to reverse a linked list.",
                "Write a JavaScript function to debounce user input.",
                "Write a Java program to implement a stack using arrays.",
                "Write a Go program to handle concurrent HTTP requests.",
                "Write a Rust function to parse JSON data."
            ],
            "general": [
                "What is the difference between TCP and UDP?",
                "How does Kubernetes manage container orchestration?",
                "What are the benefits of using a NoSQL database?",
                "Explain the SOLID principles in software design.",
                "How does a CDN improve web performance?"
            ]
        }

        category_map = {
            "explore algorithms about...": "explore",
            "how to implement...": "howto",
            "analyze this code: ": "analyze",
            "write a...": "code"
        }

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