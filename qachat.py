import streamlit as st
import pandas as pd
import bcrypt
from datetime import datetime, date
import os
import sqlite3
import requests
import json
from local_model.therapyllama import generate_response

# --- Constants ---
USERS_FILE = "users.csv"
CHAT_LOGS_FILE = "chat_logs.csv"
DB_FILE = "global.db"
API_BASE_URL = "http://127.0.0.1:5000"

EMOJI_MOODS = {
    "Joy": 0.9, "Contentment": 0.8, "Peacefulness": 0.7, "Gratitude": 0.8,
    "Hope": 0.6, "Love": 0.9, "Excitement": 0.7, "Enthusiasm": 0.7,
    "Awe": 0.6, "Interest": 0.5, "Pride": 0.6, "Amusement": 0.7,
    "Relief": 0.6, "Compassion": 0.7, "Serenity": 0.8,
    "Sadness": -0.6, "Anxiety": -0.7, "Anger": -0.8, "Fear": -0.9,
    "Guilt": -0.7, "Shame": -0.8, "Disgust": -0.7, "Frustration": -0.6,
    "Disappointment": -0.6, "Worry": -0.7, "Stress": -0.8, "Loneliness": -0.7
}

# --- DB Initialization ---
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS mood_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            email TEXT,
            datetime TEXT,
            mood TEXT,
            score REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_mood_to_db(username, email, mood):
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            "INSERT INTO mood_logs (username, email, datetime, mood, score) VALUES (?, ?, ?, ?, ?)",
            (username, email, datetime.now().isoformat(), mood, EMOJI_MOODS[mood])
        )
        conn.commit()
    except Exception as e:
        st.error(f"DB Insert Error: {e}")
    finally:
        conn.close()

# --- User Management ---
def load_users():
    try:
        return pd.read_csv(USERS_FILE, dtype={"name": str, "email": str, "password": str, "role": str})
    except FileNotFoundError:
        return pd.DataFrame(columns=["name", "email", "password", "role"])

def save_users(users_df):
    users_df.to_csv(USERS_FILE, index=False)

def signup_user(name, email, password, role):
    users = load_users()
    if email in users["email"].values:
        return False
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    new_user = pd.DataFrame([[name, email, hashed_pw, role]], columns=["name", "email", "password", "role"])
    users = pd.concat([users, new_user], ignore_index=True)
    save_users(users)
    return True

def authenticate_user(email, password):
    users = load_users()
    user = users[users["email"] == email]
    if not user.empty:
        if bcrypt.checkpw(password.encode(), user.iloc[0]["password"].encode()):
            return True, user.iloc[0]["name"], user.iloc[0]["role"]
    return False, None, None

# --- Chat Logging ---
def safe_load_chat_logs():
    if os.path.exists(CHAT_LOGS_FILE) and os.path.getsize(CHAT_LOGS_FILE) > 0:
        try:
            return pd.read_csv(CHAT_LOGS_FILE)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=["email", "timestamp", "user_message", "bot_response"])
    else:
        return pd.DataFrame(columns=["email", "timestamp", "user_message", "bot_response"])

def save_chat(email, user_msg, bot_msg):
    time = datetime.now()
    new_entry = pd.DataFrame([[email, time, user_msg, bot_msg]],
                             columns=["email", "timestamp", "user_message", "bot_response"])
    logs = safe_load_chat_logs()
    logs = pd.concat([logs, new_entry], ignore_index=True)
    logs.to_csv(CHAT_LOGS_FILE, index=False)

def delete_chat(email):
    logs = safe_load_chat_logs()
    logs = logs[logs["email"] != email]
    logs.to_csv(CHAT_LOGS_FILE, index=False)

# --- Therapy Bot Response ---
def get_bot_response(user_input):
    prompt = (
        "System: You are a therapy chatbot. Only respond to emotional or mental health-related questions. "
        "If the user asks unrelated questions (e.g., math, general knowledge, technical), refuse to answer and kindly redirect to therapy topics.\n\n"
        "User: I'm feeling really anxious lately.\n"
        "Therapist: I'm sorry to hear that. Would you like to talk more about what's causing your anxiety?\n\n"
        "User: What's 2 + 2?\n"
        "Therapist: I'm here to support your emotional and mental well-being. Let's focus on how you're feeling instead. Is something on your mind?\n\n"
        f"User: {user_input}\n"
        "Therapist:"
    )
    return generate_response(prompt)

# --- API Helper ---
def post_data(endpoint, data):
    url = f"{API_BASE_URL}{endpoint}"
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        error_message = "Network error or server unavailable."
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                error_message = error_details.get('message', e.response.reason or e.response.text)
            except json.JSONDecodeError:
                error_message = e.response.text
            if e.response.status_code == 400:
                error_message = error_details.get('message', f"Bad request: {e.response.text}")
        st.error(f"Operation failed: {error_message}")
        return None 
    except json.JSONDecodeError:
        st.error(f"Error decoding JSON response from {url}. Server response: {response.text}")
        return None

# --- Mood Entry UI ---
def mood_entry_ui():
    st.header(f"📝 Record Mood for {st.session_state.name}")
    with st.form("mood_entry_form"):
        today = date.today()
        date_selected = st.date_input("Date", value=today)
        mood_selected = st.selectbox("How are you feeling today?", list(EMOJI_MOODS.keys()))
        submitted = st.form_submit_button("Save Mood Entry")

        if submitted:
            if not st.session_state.email or not st.session_state.name:
                st.error("Session error. Please log in again.")
                return
            save_mood_to_db(st.session_state.name, st.session_state.email, mood_selected)
            st.success(f"Thanks for sharing. Mood '{mood_selected}' recorded.")

# --- Session Init ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.name = ""
    st.session_state.email = ""
    st.session_state.role = ""

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to", ["Register", "Login"])

# --- Registration Page ---
if not st.session_state.logged_in and menu == "Register":
    st.markdown("<h2 style='text-align: center;'>Register New User</h2>", unsafe_allow_html=True)
    with st.form("signup_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        role = st.selectbox("Register as", options=["patient", "therapist"])
        submitted = st.form_submit_button("Register")
        if submitted:
            if name and email and password:
                if signup_user(name, email, password, role):
                    st.success("Account created! Please log in.")
                else:
                    st.error("Email already exists.")
            else:
                st.warning("Please fill all fields.")

# --- Login Page ---
elif not st.session_state.logged_in and menu == "Login":
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if email and password:
            success, name, role = authenticate_user(email, password)
            if success:
                st.session_state.logged_in = True
                st.session_state.name = name
                st.session_state.email = email
                st.session_state.role = role
                st.rerun()
            else:
                st.error("Invalid credentials.")
        else:
            st.warning("Please enter both email and password.")

# --- Main Interface ---
if st.session_state.logged_in:
    st.title("💬 TalkBuddy")
    st.markdown(f"Welcome, **{st.session_state.name}** 👋 ({st.session_state.role})")

    _, col_logout = st.columns([2, 1])
    with col_logout:
        if st.button("Logout➡️"):
            st.session_state.logged_in = False
            st.session_state.name = ""
            st.session_state.email = ""
            st.session_state.role = ""
            st.rerun()

    # --- Patient Interface ---
    if st.session_state.role == "patient":
        mood_entry_ui()

        st.markdown("---")
        st.subheader("💬 Chat With Buddy")

        if "input_text_value" not in st.session_state:
            st.session_state.input_text_value = ""

        user_input = st.text_input("You:", key="user_input_text", value=st.session_state.input_text_value)

        col_send, col_delete = st.columns([1, 2])
        with col_send:
            if st.button("Send", key="send_button"):
                if user_input.strip():
                    with st.spinner("Buddy is typing..."):
                        bot_reply = get_bot_response(user_input)
                    save_chat(st.session_state.email, user_input, bot_reply)
                    st.session_state.input_text_value = ""
                    st.rerun()
                else:
                    st.warning("Please type a message to send.")

        with col_delete:
            if st.button("Delete My Chat History", key="delete_history_button"):
                delete_chat(st.session_state.email)
                st.success("Your chat history has been deleted.")
                st.rerun()

        st.markdown("---")
        st.subheader("📜 Your Chat History")

        logs = safe_load_chat_logs()
        user_logs = logs[logs["email"] == st.session_state.email]

        if not user_logs.empty:
            for _, row in user_logs.iloc[::-1].iterrows():
                st.markdown(
                    f"""
                    <div style='font-size:12px;color:#888;margin-bottom:5px;'>🕒 {pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}</div>
                    <div style='background-color:#f0f0f5;padding:10px;border-radius:10px;margin-bottom:5px; color:#000;'>
                        🙎🏻‍♂️ <strong>You:</strong><br><span style='font-size:15px'>{row['user_message']}</span>
                    </div>
                    <div style='background-color:#e6f2ff;padding:10px;border-radius:10px;margin-bottom:10px; color:#000;'>
                        👤 <strong>Buddy:</strong><br><span style='font-size:15px'>{row['bot_response']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No chat history found. Start chatting!")

    # --- Therapist Interface ---
    elif st.session_state.role == "therapist":
        st.info("🧑‍⚕️ This is the **Therapist Dashboard**. Features coming soon.")
