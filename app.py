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
    user_id = Column(String)  # Added to track user sessions
    user_msg = Column(Text)
    ai_msg = Column(Text)
    timestamp = Column(String)
    title = Column(String)
    welcome_shown = Column(Integer, default=0)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Database Functions
def store_chat(chat_id, user_id, user_msg, ai_msg, title=None, welcome_shown=0):
    session = Session()
    try:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        title = title or (user_msg[:50].strip() if user_msg else f"Chat {chat_id}")[:50]
        ai_msg = re.sub(r'\[INST\].*?\[/INST\]', '', ai_msg, flags=re.DOTALL)
        ai_msg = re.sub(r'<s>', '', ai_msg).strip()
        chat = ChatHistory(
            chat_id=chat_id,
            user_id=user_id,
            user_msg=user_msg,
            ai_msg=ai_msg,
            timestamp=timestamp,
            title=title,
            welcome_shown=welcome_shown
        )
        session.merge(chat)
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
        chats = session.query(ChatHistory).with_entities(ChatHistory.chat_id, ChatHistory.title).order_by(ChatHistory.timestamp.desc()).all()
        unique_titles = []
        seen_chat_ids = set()
        for chat in chats:
            if chat.chat_id in seen_chat_ids or not chat.title:
                continue
            unique_titles.append({"chat_id": chat.chat_id, "title": chat.title[:100]})
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

# Query Classification
def classify_query(prompt):
    prompt = prompt.strip().lower()
    tech_keywords = r"\b(code|python|java|c\+\+|javascript|js|typescript|ruby|php|go|rust|kotlin|swift|c#|perl|scala|r|matlab|sql|nosql|algorithm|O\(.*\)|recursion|data structure|machine learning|neural network|database|API|backend|frontend|AI|time complexity|sorting|engineering|system design|software|hardware|math|algebra|calculus|geometry|statistics|probability|optimization|cloud|devops|docker|kubernetes|git|aws|azure|gcp|ci|cd|cybersecurity|game|development|network|array)\b"
    tech_patterns = [
        r"how does .* work\??", r"how to .*", r"what is the best way to .*",
        r"compare .* vs .*", r"why is .* better than .*", r"how can .* be improved\??",
        r"build .*", r"create .*", r"implement .*", r"design .*", r"optimize .*"
    ]
    debug_patterns = [r"why does this code.*fail", r"debug this", r"fix this code", r"error in this code"]
    developer_patterns = [r"who created", r"who made", r"who developed", r"who is behind"]
    greetings = ["hi", "hello", "hey", "howdy", "greetings", "salutations", "introduce"]

    is_greeting = any(g in prompt for g in greetings)
    is_tech = re.search(tech_keywords, prompt, re.IGNORECASE) or any(re.search(p, prompt, re.IGNORECASE) for p in tech_patterns)
    is_debugging = any(re.search(p, prompt, re.IGNORECASE) for p in debug_patterns)
    is_developer = any(re.search(p, prompt, re.IGNORECASE) for p in developer_patterns)

    if is_greeting and is_tech:
        return "hybrid_greeting_tech"
    elif is_greeting and is_debugging:
        return "hybrid_greeting_debugging"
    elif is_developer:
        return "developer"
    elif is_greeting:
        return "greeting"
    elif is_tech:
        return "tech"
    elif is_debugging:
        return "debugging"
    return "general"

# Response Formatting
def format_response(response):
    response = re.sub(r'AlgoAI', 'the assistant', response, flags=re.IGNORECASE)
    response = re.sub(r'\[INST\].*?\[/INST\]', '', response, flags=re.DOTALL)
    response = re.sub(r'<s>', '', response).strip()
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
    if not re.search(r"\*\*Code Example $$       Python       $$:\*\*", response):
        response = re.sub(r"```python", "**Code Example (Python):**\n```python", response)
    if not re.search(r"\*\*Code Example $$       Java       $$:\*\*", response):
        response = re.sub(r"```java", "**Code Example (Java):**\n```java", response)
    if not re.search(r"\*\*Code Example $$       C\+\+       $$:\*\*", response):
        response = re.sub(r"```cpp", "**Code Example (C++):**\n```cpp", response)
    if not re.search(r"\*\*Code Example $$       JavaScript       $$:\*\*", response):
        response = re.sub(r"```javascript", "**Code Example (JavaScript):**\n```javascript", response)
    if not re.search(r"\*\*Code Example $$       TypeScript       $$:\*\*", response):
        response = re.sub(r"```typescript", "**Code Example (TypeScript):**\n```typescript", response)
    if not re.search(r"\*\*Code Example $$       Go       $$:\*\*", response):
        response = re.sub(r"```go", "**Code Example (Go):**\n```go", response)
    if not re.search(r"\*\*Code Example $$       Rust       $$:\*\*", response):
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
        "You are an expert coding assistant specializing in algorithms, system design, and software development, designed to help programmers of all levels understand concepts deeply before writing code. Your primary goal is to ensure users grasp the logic and reasoning behind solutions through structured, step-by-step guidance, while also providing practical, real-world examples and optimization insights. Your responses must always be:\n"
        "- Educational and Insightful: Prioritize explaining concepts, logic, and reasoning before providing code.\n"
        "- Structured and Step-by-Step: Break down problems into clear, numbered sections.\n"
        "- Professional and Technical: Assume the user is a developer or engineer, using precise terminology while remaining approachable for beginners.\n"
        "- Interactive and Engaging: Include logical suggestions, comparisons, and a specific follow-up question at the end of each response to encourage deeper exploration.\n"
        "- Context-Aware: Reference previous messages to maintain a natural conversation flow, building on past responses when relevant (e.g., 'Continuing from your last question about quicksort...'). If it’s a new chat, start fresh without referencing past messages.\n"
        "- Practical and Real-World Focused: Use real-world scenarios or use cases in explanations and code examples to make solutions applicable.\n"
        "- Optimization-Oriented: Highlight best practices, suggest cleaner or more efficient code, and address scalability considerations where relevant.\n"
        "- Chat Title Generation: When storing a chat, generate a concise, meaningful title based on the user’s first message (e.g., 'Explain Quicksort' for 'Explain quicksort in Python'). Avoid generic prefixes like 'Chat about' and limit the title to 50 characters.\n"
        "\n"
        "**Strict Response Format (for all modes):**\n"
        "- Use **Markdown formatting** to create a structured, readable response:\n"
        "  - Use **bold (**text**)** for section headers and key terms (e.g., **Concept Explanation**).\n"
        "  - Use numbered lists (e.g., 1., 2.) for step-by-step breakdowns.\n"
        "  - Use bullet points (e.g., -) for lists of items, such as alternative approaches or optimization tips.\n"
        "  - Use code blocks with language specification (e.g., ```python) for all code examples.\n"
        "- For long or detailed sections (e.g., deep dives, alternative approaches), wrap the content in a collapsible section using HTML `<details>` and `<summary>` tags (e.g., <details><summary>Alternative Approaches</summary>...</details>).\n"
        "- Ensure code blocks are clean, well-commented, and preceded by a brief explanation of what the code does.\n"
        "- End each response with a specific, relevant follow-up question in bold (e.g., **Would you like to explore...?**).\n"
        "- Do not mention any specific identity or name in responses; use generic phrasing like 'Here is an example...' or 'This approach suggests...'.\n"
        "\n"
        "**General Guidelines:**\n"
        "- Detect the user’s preferred programming language from the query (e.g., 'in Python', 'using Java') and provide examples in that language; default to Python if no language is specified.\n"
        "- Adapt the depth of your response based on the query’s complexity and user requests (e.g., a 'Deep Dive' request should provide more detailed analysis and additional examples).\n"
        "- For debugging queries, guide the user through identifying and fixing errors step-by-step, explaining the root cause and suggesting preventive measures.\n"
        "- When providing code, ensure it is clean, well-commented, and accompanied by an explanation of how it works and why it’s a good solution.\n"
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
            "Provide a warm and engaging introduction to set the tone for an educational coding journey:\n"
            "**1. Welcome Message**: Greet the user warmly, mentioning the current time of day (e.g., 'Good morning!') and introducing the assistant as a coding guide focused on understanding and problem-solving (2-3 sentences).\n"
            "**2. Capabilities**: Highlight expertise in breaking down coding problems, supporting multiple languages, and providing practical, optimized solutions (1-2 sentences).\n"
            "**3. Invitation**: Encourage the user to ask a coding-related question or explore a topic, suggesting they specify a language if desired (1 sentence)."
        )})
        max_tokens = 200
        temp = 0.5
        follow_up = "**What coding topic would you like to explore today? Feel free to specify a language!**"
    elif mode == "hybrid_greeting_tech":
        messages.append({"role": "system", "content": (
            "Handle a hybrid intent with a greeting and a technical question:\n"
            "**1. Greeting Acknowledgment**: Respond briefly to the greeting (e.g., 'Hello! Let’s dive into your question...') (1 sentence).\n"
            "**2. Technical Response**: Provide a detailed, educational response:\n"
            "   - **Clarify Intent**: Restate the user’s query, confirm the preferred language (**{preferred_language}** if specified, else Python), and outline the goal (1 sentence).\n"
            "   - **Concept Explanation**: Explain the concept or algorithm’s foundation, including its purpose and a real-world use case (2-3 sentences).\n"
            "   - **Logical Breakdown**: Break down the solution into logical steps, explaining the reasoning behind each step (3-4 sentences).\n"
            "   - **Complexity Analysis**: Analyze the time and space complexity (Big-O notation), explaining factors that affect performance (2-3 sentences).\n"
            "   - **Example Implementation**: Provide a clean, well-commented code example in **{preferred_language}** within ``` blocks, preceded by a brief explanation (3-4 sentences).\n"
            "   - **Alternative Approaches**: Compare with at least one alternative approach, wrapped in <details><summary>Alternative Approaches</summary>...</details> (2-3 sentences).\n"
            "   - **Performance Optimization**: Suggest improvements, wrapped in <details><summary>Performance Optimization</summary>...</details> (2-3 sentences).\n"
            "   - **Next Steps**: Recommend a related topic (1 sentence)."
        )})
        max_tokens = 1000
        temp = 0.2
        follow_up = "**Would you like a deeper explanation or a different approach?**"
    elif mode == "hybrid_greeting_debugging":
        messages.append({"role": "system", "content": (
            "Handle a hybrid intent with a greeting and a debugging question:\n"
            "**1. Greeting Acknowledgment**: Respond briefly to the greeting (e.g., 'Hello! Let’s fix your code...') (1 sentence).\n"
            "**2. Debugging Response**: Provide a structured debugging guide:\n"
            "   - **Identify the Issue**: Analyze the user’s code or description, pinpoint the cause, and explain why it occurs (2-3 sentences).\n"
            "   - **Step-by-Step Debugging**: Break down the debugging process into clear steps (3-4 sentences).\n"
            "   - **Fixed Code**: Provide a corrected code example within ``` blocks, explaining changes (3-4 sentences).\n"
            "   - **Preventive Tips**: Suggest best practices to avoid similar issues (1-2 sentences).\n"
            "   - **Next Steps**: Encourage testing or exploring a related topic (1 sentence)."
        )})
        max_tokens = 800
        temp = 0.3
        follow_up = "**Would you like to test the fix or explore error handling further?**"
    elif mode == "tech":
        language_match = re.search(r'\b(in|using)\s*(python|java|c\+\+|javascript|typescript|go|rust|ruby|php|kotlin|swift)\b', prompt, re.IGNORECASE)
        preferred_language = language_match.group(2).lower() if language_match else "python"
        if deep_dive and last_response:
            messages.append({"role": "system", "content": (
                "Provide a deeper analysis by building on the previous response: **Previous response: {last_response}**\n"
                "**1. In-Depth Analysis**: Analyze the previous response’s strengths, limitations, and real-world applicability (3-4 sentences).\n"
                "**2. Debugging and Improvements**: Identify edge cases or errors, suggesting fixes or optimizations (3-4 sentences).\n"
                "**3. Alternative Implementation**: Provide a new code example, explaining its benefits, wrapped in <details><summary>Alternative Implementation</summary>...</details> (3-4 sentences).\n"
                "**4. Next Steps**: Suggest a related topic or practical application (1 sentence).\n"
                "Ensure all insights are unique."
            )})
            max_tokens = 1200
            temp = 0.2
            follow_up = "**Would you like a different approach or more depth?**"
        else:
            messages.append({"role": "system", "content": (
                "Provide a detailed, educational response to help the user understand and apply the solution:\n"
                "**1. Clarify Intent**: Restate the query, confirm the preferred language (**{preferred_language}**), and outline the goal (1 sentence).\n"
                "**2. Concept Explanation**: Explain the concept, including a real-world use case (2-3 sentences).\n"
                "**3. Logical Breakdown**: Break down the solution into steps, explaining reasoning (3-4 sentences).\n"
                "**4. Complexity Analysis**: Analyze time and space complexity, explaining factors (2-3 sentences).\n"
                "**5. Example Implementation**: Provide a clean, commented code example in **{preferred_language}** within ``` blocks, with an explanation (3-4 sentences).\n"
                "**6. Alternative Approaches**: Compare with an alternative, wrapped in <details><summary>Alternative Approaches</summary>...</details> (2-3 sentences).\n"
                "**7. Performance Optimization**: Suggest improvements, wrapped in <details><summary>Performance Optimization</summary>...</details> (2-3 sentences).\n"
                "**8. Next Steps**: Recommend a related topic (1 sentence)."
            )})
            max_tokens = 1000
            temp = 0.2
            follow_up = "**Would you like a deeper explanation?**"
    elif mode == "debugging":
        messages.append({"role": "system", "content": (
            "Provide a structured debugging guide:\n"
            "**1. Identify the Issue**: Analyze the code or description, pinpoint the cause, and explain why it occurs (2-3 sentences).\n"
            "**2. Step-by-Step Debugging**: Break down the debugging process into clear steps (3-4 sentences).\n"
            "**3. Fixed Code**: Provide a corrected code example within ``` blocks, explaining changes (3-4 sentences).\n"
            "**4. Preventive Tips**: Suggest best practices to avoid similar issues (1-2 sentences).\n"
            "**5. Next Steps**: Encourage testing or exploring a related topic (1 sentence)."
        )})
        max_tokens = 800
        temp = 0.3
        follow_up = "**Would you like to test the fix or explore error handling further?**"
    elif mode == "developer":
        messages.append({"role": "system", "content": (
            "Provide a detailed introduction about the creators of this assistant:\n"
            "**1. Team Overview**: Introduce the Algo Team as the creators of this coding assistant, emphasizing its purpose and development journey (2-3 sentences).\n"
            "**2. Key Members**: Highlight the key contributors, SYED RAYAN and SHAIK AYUB, with details about their roles and contributions (3-4 sentences).\n"
            "**3. Vision and Goals**: Explain the vision behind AlgoAI, including its focus on education, practical coding solutions, and future plans (2-3 sentences).\n"
            "**4. Next Steps**: Suggest a related topic or question for the user to explore (1 sentence)."
        )})
        max_tokens = 500
        temp = 0.2
        follow_up = "**Would you like to know more about the development process or try a coding challenge?**"
    else:
        if deep_dive and last_response:
            messages.append({"role": "system", "content": (
                "Expand on the previous response with deeper insights: **Previous response: {last_response}**\n"
                "**1. In-Depth Analysis**: Dive into mechanics, challenges, or performance (3-4 sentences).\n"
                "**2. Practical Example or Optimization**: Provide an example or optimization strategy (3-4 sentences).\n"
                "**3. Next Steps**: Suggest a hands-on activity or topic (1 sentence).\n"
                "Ensure insights are unique."
            )})
            max_tokens = 600
            temp = 0.3
            follow_up = "**Would you like to dig deeper?**"
        else:
            if "joke" in prompt.lower():
                messages.append({"role": "system", "content": (
                    "Provide a lighthearted joke, followed by a brief response related to the query (2-3 sentences)."
                )})
                follow_up = "**Do you have further questions or want a deeper explanation?**"
            else:
                messages.append({"role": "system", "content": (
                    "Provide an educational response:\n"
                    "**1. Direct Answer**: Answer clearly, defining the concept and purpose (1-2 sentences).\n"
                    "**2. Real-World Context**: Provide a practical example or use case (1-2 sentences).\n"
                    "**3. Key Insight**: Share a challenge or optimization tip (1-2 sentences).\n"
                    "**4. Next Steps**: Suggest a related topic (1 sentence)."
                )})
                follow_up = "**Do you have further questions or want a deeper explanation?**"
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
        user_id = request.headers.get("X-User-ID") or request.remote_addr  # Example user tracking
        deep_dive = data.get("deep_dive", False)
        groq_response = query_groq(chat_id, user_query, deep_dive)
        formatted_response = format_response(groq_response)
        store_chat(chat_id, user_id, user_query, formatted_response)
        return jsonify({"response": formatted_response, "chat_id": chat_id})
    except Exception as e:
        print(f"Error in get_response: {e}")
        return jsonify({"error": f"API request failed—{str(e)}", "chat_id": chat_id}), 500

@app.route("/new_chat", methods=["POST"])
def new_chat():
    try:
        user_id = request.headers.get("X-User-ID") or request.remote_addr  # Example user tracking
        session = Session()
        last_chat = session.query(ChatHistory).filter_by(user_id=user_id).order_by(ChatHistory.timestamp.desc()).first()
        session.close()

        if last_chat and not request.args.get("force_new", False):
            return jsonify({"chat_id": last_chat.chat_id, "greeting": "Welcome back! Continuing your last session..."})

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
        user_id = request.headers.get("X-User-ID") or request.remote_addr
        store_chat(chat_id, user_id, user_msg, ai_msg, title)
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
            chat.title = new_title[:50]  # Limit title to 50 characters
            session.commit()
            return jsonify({"message": "Chat title updated successfully"})
        return jsonify({"error": "Chat not found"}), 404
    except Exception as e:
        print(f"Error in update_chat_title: {e}")
        session.rollback()
        return jsonify({"error": f"Title update failed—{e}"}), 500
    finally:
        session.close()

@app.route("/clear_chats", methods=["POST"])
def clear_chats():
    session = Session()
    try:
        user_id = request.headers.get("X-User-ID") or request.remote_addr
        session.query(ChatHistory).filter_by(user_id=user_id).delete()
        session.commit()
        return jsonify({"message": "All chat history cleared successfully."})
    except Exception as e:
        print(f"Error in clear_chats: {e}")
        session.rollback()
        return jsonify({"error": f"Failed to clear chats—{e}"}), 500
    finally:
        session.close()

@app.route("/delete_chat", methods=["POST"])
def delete_chat():
    session = Session()
    try:
        data = request.get_json()
        chat_id = data.get("chat_id")
        if not chat_id:
            return jsonify({"error": "No chat_id provided."}), 400
        session.query(ChatHistory).filter_by(chat_id=chat_id).delete()
        session.commit()
        return jsonify({"message": f"Chat {chat_id} deleted successfully.", "chat_id": chat_id})
    except Exception as e:
        print(f"Error in delete_chat: {e}")
        session.rollback()
        return jsonify({"error": f"Delete failed—{e}"}), 500
    finally:
        session.close()

@app.route("/suggestions", methods=["GET"])
def get_suggestions():
    try:
        category = request.args.get("category", "general").lower()
        current_time = int(time.time())

        suggestion_pools = {
            "explore": [
                "What are the key differences between merge sort and quicksort in terms of performance and use cases?",
                "Explain the time complexity of binary search in detail with a real-world example.",
                "How does a hash table work under the hood in a database system?",
                "What is the best algorithm for graph traversal in a dense graph like a social network?",
                "Compare depth-first search and breadth-first search for tree traversal in a file system."
            ],
            "howto": [
                "How to implement a binary search tree in Python for a dictionary app?",
                "How to create a REST API using Flask for a web service?",
                "How to optimize a SQL query for a large e-commerce database?",
                "How to set up a CI/CD pipeline with GitHub Actions for a software project?",
                "How to implement authentication in a Node.js app for user login?"
            ],
            "analyze": [
                "Analyze this code for potential memory leaks: def factorial(n): return 1 if n == 0 else n * factorial(n-1)",
                "Analyze the performance of a bubble sort implementation for a small dataset.",
                "Analyze this SQL query for efficiency: SELECT * FROM users WHERE age > 30;",
                "Analyze the memory usage of a recursive Fibonacci function in a real application.",
                "Analyze the scalability of a microservices architecture for a cloud platform."
            ],
            "code": [
                "Write a Python function to reverse a linked list for a playlist manager.",
                "Write a JavaScript function to debounce user input for a search bar.",
                "Write a Java program to implement a stack using arrays for a calculator.",
                "Write a Go program to handle concurrent HTTP requests for a web server.",
                "Write a Rust function to parse JSON data for a configuration file."
            ],
            "general": [
                "What is the difference between TCP and UDP in a networked application?",
                "How does Kubernetes manage container orchestration in a cloud environment?",
                "What are the benefits of using a NoSQL database for a social media platform?",
                "Explain the SOLID principles in software design with a practical example.",
                "How does a CDN improve web performance for a global audience?"
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