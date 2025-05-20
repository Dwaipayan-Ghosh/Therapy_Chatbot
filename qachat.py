import streamlit as st
import pandas as pd
import bcrypt
from datetime import datetime
from text_classification import get_sentiment_label
from local_model.therapyllama import generate_response

USERS_FILE = "users.csv"
CHAT_LOGS_FILE = "chat_logs.csv"

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

def save_chat(username, user_msg, bot_msg):
    user_sentiment = get_sentiment_label(user_msg)
    bot_sentiment = get_sentiment_label(bot_msg)
    time = datetime.now()
    new_entry = pd.DataFrame([[username, time, user_msg, user_sentiment, bot_msg, bot_sentiment]],
                             columns=["username", "timestamp", "user_message", "user_sentiment", "bot_response", "bot_sentiment"])
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
    prompt = f"You are a kind and empathetic therapist. A user says: \"{user_input}\". How would you respond?"
    return generate_response(prompt)

def sentiment_color(sentiment):
    if sentiment.lower() == "positive":
        return "green"
    elif sentiment.lower() == "negative":
        return "red"
    else:
        return "#d4a017"  # Neutral yellow

# Session management
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

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

if st.session_state.logged_in:
    st.title("🧠 Therapy Chatbot")
    st.markdown(f"Welcome, **{st.session_state.username}** 👋")

    user_input = st.text_input("You:", key="input")

    if st.button("Send"):
        if user_input.strip():
            bot_reply = get_bot_response(user_input)
            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**Therapist:** {bot_reply}")
            save_chat(st.session_state.username, user_input, bot_reply)

    if st.button("Delete My Chat History"):
        delete_chat(st.session_state.username)
        st.success("Your chat history has been deleted.")

    try:
        logs = pd.read_csv(CHAT_LOGS_FILE)
        user_logs = logs[logs["username"] == st.session_state.username]
        if not user_logs.empty:
            st.markdown("---")
            st.subheader("📜 Your Chat History")
            for _, row in user_logs.iterrows():
                st.markdown(f"**🕒 {row['timestamp']}**")
                st.markdown(
                    f"<div style='color:{sentiment_color(row['user_sentiment'])}'>😐 <strong>You:</strong> {row['user_message']} <em>(Sentiment: {row['user_sentiment']})</em></div>",
                    unsafe_allow_html=True)
                st.markdown(
                    f"<div style='color:{sentiment_color(row['bot_sentiment'])}'>🤖 <strong>Therapist:</strong> {row['bot_response']} <em>(Sentiment: {row['bot_sentiment']})</em></div><hr>",
                    unsafe_allow_html=True)
    except FileNotFoundError:
        st.info("No chat history found.")
