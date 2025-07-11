import streamlit as st
import pandas as pd
import bcrypt
from datetime import datetime, date, timedelta
import os
import sqlite3
import requests
import json
from local_model.therapyllama import generate_response

# ─────────────────────────────
# Constants
# ─────────────────────────────
USERS_FILE = "users.csv"
CHAT_LOGS_FILE = "chat_logs.csv"
MOOD_LOGS_FILE = "mood.csv"
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

# Define keyword to URL mappings
KEYWORD_REDIRECTS = {
    "stressed": "https://open.spotify.com/playlist/6RtESsqB6n63ZVrvTfbXY1?si=hJQ8IB5PS7eazhQcSiYgLw",
    "depressed": "https://open.spotify.com/playlist/6RtESsqB6n63ZVrvTfbXY1?si=hJQ8IB5PS7eazhQcSiYgLw",
    "frustrated": "https://open.spotify.com/playlist/6RtESsqB6n63ZVrvTfbXY1?si=hJQ8IB5PS7eazhQcSiYgLw",
    "motivation": "https://www.youtube.com/watch?si=6o5uoOFCMedfOLwb&v=5dUstLhhVEI&feature=youtu.be",
    "yoga": "https://www.youtube.com/watch?si=b-FSSZgI5rwkUcID&v=eUDTHzCw5YE&feature=youtu.be",
}

SPOTIFY_PLAYLIST_URL = "https://open.spotify.com/playlist/6RtESsqB6n63ZVrvTfbXY1?si=hJQ8IB5PS7eazhQcSiYgLw"
BLINKIT_URL = "https://www.blinkit.com/"


# ─────────────────────────────
# CSV Mood Logging Functions (Replaced SQLite)
# ─────────────────────────────
def safe_load_mood_logs():
    """Safely loads mood logs from mood.csv, ensuring 'datetime' column is proper datetime type.
    Handles missing or empty files by creating an appropriately typed empty DataFrame.
    """
    # Define the columns and their desired dtypes for an empty DataFrame
    columns_and_dtypes = {
        "username": str,
        "email": str,
        "datetime": 'datetime64[ns]', # Explicitly set for datetime
        "mood": str,
        "score": float
    }
    
    if os.path.exists(MOOD_LOGS_FILE) and os.path.getsize(MOOD_LOGS_FILE) > 0:
        try:
            df = pd.read_csv(MOOD_LOGS_FILE)
            # Explicitly convert 'datetime' column to datetime objects
            # 'errors='coerce'' will turn unparseable dates into NaT (Not a Time)
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            return df
        except pd.errors.EmptyDataError:
            # File exists but is empty
            st.info(f"'{MOOD_LOGS_FILE}' exists but is empty. Initializing empty DataFrame with correct dtypes.")
            return pd.DataFrame(columns=columns_and_dtypes.keys()).astype(columns_and_dtypes)
        except Exception as e:
            # Catch other potential errors during file reading
            st.error(f"Error loading '{MOOD_LOGS_FILE}': {e}. Initializing empty DataFrame.")
            return pd.DataFrame(columns=columns_and_dtypes.keys()).astype(columns_and_dtypes)
    else:
        # File does not exist, create an empty DataFrame with proper dtypes
        st.info(f"'{MOOD_LOGS_FILE}' not found. Initializing empty DataFrame with correct dtypes.")
        return pd.DataFrame(columns=columns_and_dtypes.keys()).astype(columns_and_dtypes)

def save_mood_entry_to_csv(username, email, mood, entry_date: date):
    """
    Saves a new mood entry to mood.csv. Prevents duplicate entries for the same user on the same date.
    """
    current_mood_logs = safe_load_mood_logs()
    
    # Filter out rows where 'datetime' could not be parsed (NaT values) before checking for duplicates
    valid_mood_logs = current_mood_logs.dropna(subset=['datetime'])

    # Check for existing mood entry for this user on this specific date
    if not valid_mood_logs.empty:
       # Create 'date_only' column on the valid subset for comparison
       valid_mood_logs['date_only'] = valid_mood_logs['datetime'].dt.date
       if ((valid_mood_logs['email'] == email) & (valid_mood_logs['date_only'] == entry_date)).any():
            st.warning(f"You have already recorded a mood for {entry_date}. Please select another date or consider editing an existing entry.")
            return False # Indicate that save was not performed
    
    # If no existing entry, proceed with insertion
    # Combine selected date with current time to form a full timestamp
    full_datetime_to_save = datetime.combine(entry_date, datetime.now().time())
    
    new_entry_df = pd.DataFrame([{
        "username": username,
        "email": email,
        "datetime": full_datetime_to_save, # Store as datetime object, pandas will save as ISO string
        "mood": mood,
        "score": EMOJI_MOODS[mood]
    }])
    
    # Reconstruct the full DataFrame by concatenating the valid_mood_logs and the new entry
    # and then saving. Ensure datetime column from new_entry_df is also proper datetime type.
    updated_mood_logs = pd.concat([valid_mood_logs.drop(columns=['date_only']) if 'date_only' in valid_mood_logs.columns else valid_mood_logs, new_entry_df], ignore_index=True)
    
    updated_mood_logs.to_csv(MOOD_LOGS_FILE, index=False)
    st.success(f"Thanks for sharing. Mood '{mood}' recorded for {entry_date}.")
    return True # Indicate successful save

# ─────────────────────────────
# User Management (unchanged, uses users.csv)
# ─────────────────────────────
def load_users():
    try:
        # Ensure 'role' is read as string to prevent issues with mixed types
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

def reset_user_password(email, new_password):
    users = load_users()
    if email not in users["email"].values:
        return False
    
    hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    users.loc[users["email"] == email, "password"] = hashed_pw
    save_users(users)
    return True

# ─────────────────────────────
# Chat Logging (unchanged, uses chat_logs.csv)
# ─────────────────────────────
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

# ─────────────────────────────
# Therapy Bot Response (MODIFIED)
# ─────────────────────────────
def get_bot_response(user_input):
    user_input_lower = user_input.lower()

    # Check for specific keyword redirects
    for keyword, url in KEYWORD_REDIRECTS.items():
        if keyword in user_input_lower:
            return f"It sounds like you're looking for some support. Here's a resource that might help: [Click here]({url})"
    
    # Check for "song" keyword
    if "song" in user_input_lower:
        return f"Music can be a great way to express yourself. Here's a Spotify playlist you might enjoy: [Spotify Playlist]({SPOTIFY_PLAYLIST_URL})"

    # Check for "food" keyword
    if "food" in user_input_lower:
        return f"Taking care of your physical needs is important. If you're looking for food delivery, you can check out: [Blinkit]({BLINKIT_URL})"


    # Original chatbot logic for therapy-related questions
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

# ─────────────────────────────
# API Helper (unchanged - if you use an external API)
# ─────────────────────────────
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

# ─────────────────────────────
# Mood Entry UI (Updated for CSV saving)
# ─────────────────────────────
def mood_entry_ui():
    st.header(f"📝 Record Mood for {st.session_state.name}")
    with st.form("mood_entry_form"):
        today = date.today()
        # Allow selecting date, default to today, and restrict to today or past
        date_selected = st.date_input("Select Date for Mood Entry", value=today, max_value=today) 
        mood_selected = st.selectbox("How are you feeling on this date?", list(EMOJI_MOODS.keys()))
        submitted = st.form_submit_button("Save Mood Entry")

        if submitted:
            if not st.session_state.email or not st.session_state.name:
                st.error("Session error. Please log in again.")
                return
            # Call the new CSV saving function
            save_mood_entry_to_csv(st.session_state.name, st.session_state.email, mood_selected, date_selected)


# ─────────────────────────────
# Session & Navigation
# ─────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.name = ""
    st.session_state.email = ""
    st.session_state.role = ""

# Keep track of the current page for navigation
if "page" not in st.session_state:
    st.session_state.page = "Login"

st.sidebar.title("Navigation")
if not st.session_state.logged_in:
    # Ensure all possible non-logged-in pages are in menu_options when needed
    menu_options = ["Register", "Login", "Forgot Password"]
    if st.session_state.page == "Reset Password":
        # If currently on Reset Password page, don't show it in radio, keep current
        # or handle navigation specifically to prevent errors if it's not a selectable menu item
        pass 
    else:
        selected_menu = st.sidebar.radio("Go to", menu_options, index=menu_options.index(st.session_state.page))
        st.session_state.page = selected_menu
else:
    # If logged in, only offer Logout in sidebar
    st.sidebar.markdown(f"Logged in as **{st.session_state.name}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.name = ""
        st.session_state.email = ""
        st.session_state.role = ""
        st.session_state.page = "Login" # Redirect to login after logout
        st.rerun()


# ─────────────────────────────
# Registration Page
# ─────────────────────────────
if not st.session_state.logged_in and st.session_state.page == "Register":
    st.markdown("<h2 style='text-align: center;'>Register New User</h2>", unsafe_allow_html=True)
    with st.form("signup_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Register")
        if submitted:
            if name and email and password:
                if signup_user(name, email, password ,"patient"): # Role is hardcoded as "patient"
                    st.success("Account created! Please log in.")
                    st.session_state.page = "Login" # Redirect to login after successful registration
                    st.rerun()
                else:
                    st.error("Email already exists.")
            else:
                st.warning("Please fill all fields.")

# ─────────────────────────────
# Login Page
# ─────────────────────────────
elif not st.session_state.logged_in and st.session_state.page == "Login":
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    
    col_login, col_forgot_password = st.columns([1, 1])
    with col_login:
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
    with col_forgot_password:
        if st.button("Forgot Password?"):
            st.session_state.page = "Forgot Password"
            st.rerun()

# ─────────────────────────────
# Forgot Password Page
# ─────────────────────────────
elif not st.session_state.logged_in and st.session_state.page == "Forgot Password":
    st.title("Forgot Password")
    st.warning("⚠️ **Security Warning:** In a real application, password reset would involve sending an email with a unique link for verification. This simplified version directly allows password change for demonstration purposes and is **not secure for production environments**.")
    
    forgot_email = st.text_input("Enter your registered email", key="forgot_email")
    if st.button("Proceed to Reset"):
        users_df = load_users()
        if forgot_email in users_df["email"].values:
            st.session_state.forgot_password_email = forgot_email
            st.session_state.page = "Reset Password"
            st.rerun()
        else:
            st.error("Email not found. Please check your email address.")

# ─────────────────────────────
# Reset Password Page
# ─────────────────────────────
elif not st.session_state.logged_in and st.session_state.page == "Reset Password":
    if "forgot_password_email" not in st.session_state:
        st.error("No email provided for password reset. Please go back to Forgot Password.")
        if st.button("Go to Forgot Password"):
            st.session_state.page = "Forgot Password"
            st.rerun()
    else:
        st.title(f"Reset Password for {st.session_state.forgot_password_email}")
        new_password = st.text_input("Enter new password", type="password", key="new_password")
        confirm_password = st.text_input("Confirm new password", type="password", key="confirm_password")
        
        if st.button("Reset Password"):
            if new_password and confirm_password:
                if new_password == confirm_password:
                    if reset_user_password(st.session_state.forgot_password_email, new_password):
                        st.success("Your password has been reset successfully! Please log in with your new password.")
                        # Clear the temporary email and redirect to login
                        del st.session_state.forgot_password_email
                        st.session_state.page = "Login"
                        st.rerun()
                    else:
                        st.error("Failed to reset password. Please try again.")
                else:
                    st.error("Passwords do not match.")
            else:
                st.warning("Please enter and confirm your new password.")

# ─────────────────────────────
# Main Interface
# ─────────────────────────────
if st.session_state.logged_in:
    st.title("💬 CureMate")
    st.markdown(f"Welcome, **{st.session_state.name}** 👋 ({st.session_state.role})")

    # --- Patient Interface ---
    if st.session_state.role == "patient":
        mood_entry_ui()

        st.markdown("---")
        st.subheader("💬 Chat With Your Mate")

        if "input_text_value" not in st.session_state:
            st.session_state.input_text_value = ""

        user_input = st.text_input("You:", key="user_input_text", value=st.session_state.input_text_value)

        col_send, col_delete = st.columns([1, 2])
        with col_send:
            if st.button("Send", key="send_button"):
                if user_input.strip():
                    with st.spinner("Your Mate is typing..."):
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
        st.info("🧑‍⚕️ This is the **Therapist Dashboard**. Please open `therapist.py` in a separate Streamlit process to access therapist features.")