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
        title = title or (user_msg[:50].strip() if user_msg else f"Chat {chat_id}")
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
        history = [{"user": chat.user_msg, "ai": chat.ai_msg} for chat in chats if chat.user_msg or chat.ai_msg]
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
            if chat.chat_id in seen_chat_ids:
                continue
            if not chat.user_msg:
                continue
            final_title = (chat.title if chat.title and chat.title.strip() else
                          (chat.user_msg[:50].strip() if chat.user_msg and chat.user_msg.strip() else
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
    response = re.sub(r'AlgoAI', 'the assistant', response, flags=re.IGNORECASE)
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
    chat_history = get_chat_history(chat_id)[-MAX_HISTORY:]

    SYSTEM_PROMPT = (
        "You are an expert in coding, algorithms, and system design. Your responses must always be:\n"
        "- **Fully detailed and well-structured** with clear, numbered sections as instructed.\n"
        "- **Professional & technical**, assuming the user is a developer or engineer.\n"
        "- **Highly interactive** with logical suggestions, improvements, and follow-up questions.\n"
        "- **Context-aware**, referencing previous messages to maintain a natural flow.\n"
        "- **Focused on best practices, optimization, and scalability considerations.**\n\n"
        "**Strict Response Format (for all modes):**\n"
        "- Use **bold (**text**) for key terms and section headers.\n"
        "- End each response with a specific follow-up question as instructed.\n"
        "- Do not repeat previous responses unless explicitly building on them.\n"
        "- Do not mention any specific identity or name (e.g., 'AlgoAI') in responses; use generic phrasing like 'Here is an example...' or 'This approach suggests...'\n\n"
        "**Mode-Specific Rules:**\n"
        "- For tech queries, include all sections: Concept Explanation, Complexity Analysis, Example Implementation, Alternative Approaches, Performance Optimization, Next Steps.\n"
        "- For general queries or deep dives, adapt the structure but maintain depth and interactivity.\n"
        "- If a code example is requested but not provided, include a default example (e.g., a simple Python `print()` statement).\n"
        "- For developer queries (e.g., 'who developed'), respond with: 'This application was developed by the Algo Team.'"
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
            "Provide a detailed and engaging introduction:\n"
            "**1. Welcome Message**: Greet the user warmly and introduce the assistant's purpose (2-3 sentences).\n"
            "**2. Capabilities**: Highlight key areas of expertise (e.g., coding, algorithms, system design) (1-2 sentences).\n"
            "**3. Invitation**: Encourage the user to ask a question (1 sentence)."
        )})
        max_tokens = 200
        temp = 0.5
        follow_up = "**What would you like to explore today?**"
    elif mode == "tech":
        language_match = re.search(r'\b(in|using)\s*(python|java|c\+\+|javascript|typescript|go|rust|ruby|php|kotlin|swift)\b', prompt, re.IGNORECASE)
        preferred_language = language_match.group(2).lower() if language_match else "python"
        if deep_dive and last_response:
            messages.append({"role": "system", "content": (
                f"Build on the previous response: **Previous response: {last_response}**\n"
                "**1. Detailed Analysis**: Analyze the last response in depth (2-3 sentences).\n"
                "**2. Improvements**: Suggest specific optimizations or debugging tips if code was included (2-3 sentences).\n"
                "**3. Alternatives**: Compare with at least one alternative approach (1-2 sentences).\n"
                "**4. Next Steps**: Propose a related topic or question (1 sentence).\n"
                "No repeats—keep it new."
            )})
            max_tokens = 1200
            temp = 0.2
            follow_up = "**Would you like a different approach or more depth?**"
        else:
            messages.append({"role": "system", "content": (
                f"Provide a clear, step-by-step technical response:\n"
                f"**1. Clarify Intent**: State the intent and preferred language (**{preferred_language}** if specified, else Python) (1 sentence).\n"
                "**2. Concept Explanation**: Explain the algorithmic foundation with theory (2-3 sentences).\n"
                "**3. Complexity Analysis**: Break down the logical structure with Big-O time and space complexity (2-3 sentences).\n"
                "**4. Example Implementation**: Show a clean, well-commented code example in {preferred_language} with ``` blocks (2-3 sentences).\n"
                "**5. Alternative Approaches**: Suggest at least one alternative solution (1-2 sentences).\n"
                "**6. Performance Optimization**: Offer ways to improve efficiency or scalability (1-2 sentences).\n"
                "**7. Next Steps**: Encourage exploration of related topics (1 sentence)."
            )})
            max_tokens = 1000
            temp = 0.2
            follow_up = "**Would you like a deeper explanation?**"
    elif mode == "developer":
        return "This application was developed by the Algo Team.\n\n**Do you have any other questions about the application?**"
    else:
        if deep_dive and last_response:
            messages.append({"role": "system", "content": (
                f"Expand on the previous response: **Previous response: {last_response}**\n"
                "**1. In-Depth Analysis**: Provide detailed insights or examples (2-3 sentences).\n"
                "**2. Improvements**: Suggest enhancements or related considerations (1-2 sentences).\n"
                "**3. Next Steps**: Encourage further exploration (1 sentence).\n"
                "No repeats—keep it new."
            )})
            max_tokens = 600
            temp = 0.3
            follow_up = "**Would you like to dig deeper?**"
        else:
            if "joke" in prompt.lower():
                messages.append({"role": "system", "content": (
                    "Provide a lighthearted joke, followed by a brief response related to the query (2-3 sentences)."
                )})
                follow_up = "**Do you have further questions or would you like a deeper explanation on a topic?**"
            else:
                messages.append({"role": "system", "content": (
                    "Give a clear, concise response:\n"
                    "**1. Direct Answer**: Answer the question directly (1-2 sentences).\n"
                    "**2. Context**: Add relevant context if applicable (1 sentence)."
                )})
                follow_up = "**Do you have further questions or would you like a deeper explanation on a topic?**"
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
            if follow_up and follow_up not in bot_response:
                bot_response += f"\n\n{follow_up}"
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
    return jsonify({"message": "Welcome to your coding assistant! Use /query to begin."}), 200

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
        # Store the chat only when a user message is sent
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
            "Welcome to your coding assistant! 🚀 I'm here to help with coding, algorithms, and system design. What would you like to explore?",
            "Greetings! I'm your technical assistant for coding and design challenges. What topic can I assist you with today?",
            "Hello! I'm here to dive into the world of coding and algorithms with you. What's your first question?",
            "Welcome back! I'm ready to assist with your next coding challenge. What would you like to explore?"
        ]
        existing_chats = get_all_chats()
        is_returning = len(existing_chats) > 0
        greeting = random.choice(welcome_messages)
        if is_returning:
            greeting = "Welcome back! I'm ready to assist with your next coding challenge. What would you like to explore?"
        # Do not store the chat in the database yet; wait for the first user message
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
        title = get_chat_title(chat_id) or (history[0]["user"][:50].strip() if history and history[0]["user"] else f"Chat #{chat_id}")
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
        current_time = int(time.time())  # Use current timestamp for randomization

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