import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import streamlit as st

# Load model and tokenizer locally
model = AutoModelForCausalLM.from_pretrained("./local_model", torch_dtype=torch.float16).to("cuda")

tokenizer = AutoTokenizer.from_pretrained("./local_model")
model.eval()

# Text generation function
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit app
st.set_page_config(page_title="Therapy Chatbot")
st.header("🧠 Therapy Chatbot")

user_input = st.text_input("How are you feeling today?")
submit = st.button("Talk")

if submit and user_input:
    response = generate_response(user_input)
    st.write(response)
