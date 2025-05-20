import streamlit as st
from llama_cpp import Llama
import pandas as pd
import os
from datetime import datetime

# Load GGUF model
@st.cache_resource
def load_model():
    return Llama(
        model_path="local_model\TherapyLlama-8B-v1-Q4_K_M.gguf",
        n_ctx=2048,
        n_threads=8,     # Use your CPU threads
        n_batch=256,
        verbose=False
    )

llm = load_model()

# Initialize CSV
CSV_FILE = "chat_logs.csv"
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["timestamp", "user", "bot"]).to_csv(CSV_FILE, index=False)

# Streamlit UI
st.title("🧠 Therapy Chatbot")
st.markdown("Talk to a local therapy chatbot powered by **TherapyLlama 8B**")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat input
user_input = st.text_input("You:", key="input")

if user_input:
    # Format prompt
    prompt = f"User: {user_input}\nTherapist:"
    
    # Generate response
    response = llm(prompt, max_tokens=512, stop=["User:", "Therapist:"], echo=False)
    bot_output = response["choices"][0]["text"].strip()

    # Display chat
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Therapist", bot_output))

    # Save to CSV
    chat_df = pd.read_csv(CSV_FILE)
    chat_df.loc[len(chat_df)] = [datetime.now(), user_input, bot_output]
    chat_df.to_csv(CSV_FILE, index=False)

# Display conversation
for role, msg in st.session_state.chat_history:
    st.markdown(f"**{role}:** {msg}")
