import streamlit as st
import pandas as pd
import bcrypt
from datetime import datetime
from local_model.therapyllama import generate_response

USERS_FILE = "users.csv"
CHAT_LOGS_FILE = "chat_logs.csv"

# --- User Management ---
def load_users():
    try:
        return pd.read_csv(USERS_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["username", "password"])

def save_users(users_df):
    users_df.to_csv(USERS_FILE, index=False)

def signup_user(username, password):
    users = load_users()
    if username in users["username"].values:
        return False
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    new_user = pd.DataFrame([[username, hashed_pw]], columns=["username", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    save_users(users)
    return True

def authenticate_user(username, password):
    users = load_users()
    user = users[users["username"] == username]
    if not user.empty:
        return bcrypt.checkpw(password.encode(), user.iloc[0]["password"].encode())
    return False

# --- Chat Logging ---
def save_chat(username, user_msg, bot_msg):
    time = datetime.now()
    new_entry = pd.DataFrame([[username, time, user_msg, bot_msg]],
                             columns=["username", "timestamp", "user_message", "bot_response"])
    try:
        logs = pd.read_csv(CHAT_LOGS_FILE)
        logs = pd.concat([logs, new_entry], ignore_index=True)
    except FileNotFoundError:
        logs = new_entry
    logs.to_csv(CHAT_LOGS_FILE, index=False)

def delete_chat(username):
    try:
        logs = pd.read_csv(CHAT_LOGS_FILE)
        logs = logs[logs["username"] != username]
        logs.to_csv(CHAT_LOGS_FILE, index=False)
    except FileNotFoundError:
        pass
def get_bot_response(user_input):
    # Construct a system-style instruction and dialogue history
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


# --- Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# --- Login / Signup UI ---
if not st.session_state.logged_in:
    st.sidebar.title("Login / Signup")
    menu = st.sidebar.radio("Menu", ["Login", "Signup"])

    if menu == "Signup":
        st.title("Create Account")
        new_user = st.text_input("New Username")
        new_pass = st.text_input("New Password", type="password")
        if st.button("Signup"):
            if signup_user(new_user, new_pass):
                st.success("Account created! Please login.")
            else:
                st.error("Username already exists.")
    else:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials.")

# --- Main Chat UI ---
if st.session_state.logged_in:
    st.title("🧠 Therapy Chatbot")
    st.markdown(f"Welcome, **{st.session_state.username}** 👋")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

    user_input = st.text_input("You:", key="input")

    if st.button("Send"):
        if user_input.strip():
            bot_reply = get_bot_response(user_input)
            st.session_state.last_input = user_input
            st.session_state.last_reply = bot_reply
            save_chat(st.session_state.username, user_input, bot_reply)
            st.rerun()

    if st.button("Delete My Chat History"):
        delete_chat(st.session_state.username)
        st.success("Your chat history has been deleted.")

    # Display chat history
    try:
        logs = pd.read_csv(CHAT_LOGS_FILE)
        user_logs = logs[logs["username"] == st.session_state.username]
        if not user_logs.empty:
            st.markdown("---")
            st.subheader("📜 Your Chat History")
            for _, row in user_logs.iterrows():
                st.markdown(
                    f"""
                    <div style='background-color:#f0f0f5;padding:10px;border-radius:10px;margin-bottom:5px; color:#000;'>
                        👤 <strong>You:</strong><br><span style='font-size:15px'>{row['user_message']}</span>
                    </div>
                    <div style='background-color:#e6f2ff;padding:10px;border-radius:10px;margin-bottom:10px; color:#000;'>
                        🤖 <strong>Therapist:</strong><br><span style='font-size:15px'>{row['bot_response']}</span>
                    </div>
                    <div style='font-size:12px;color:#888;margin-bottom:15px;'>🕒 {row['timestamp']}</div>
                    """,
                    unsafe_allow_html=True
                )
    except FileNotFoundError:
        st.info("No chat history found.")
