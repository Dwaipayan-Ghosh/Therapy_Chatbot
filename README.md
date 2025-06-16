# CureMate 💬🧠  
*AI-Powered Companion for Mental Health Monitoring and Support*  

---

## 📌 Overview

**CureMate** is a dual-user AI-powered mental health web application that provides:

- 🤖 A therapy-focused chatbot for **patients**
- 📊 Mood trend analysis and emotional forecasting for **therapists**

It bridges the gap between therapy sessions using mood tracking, conversational AI, and intelligent analytics.

---

## 🧠 Features

### 👥 Dual Role Access
- **Patients**: Mood logging, chatbot interaction, personal mood history.
- **Therapists**: View assigned patients, track mood trends, access analytics & forecasts.

### 💬 AI Chatbot
- Fine-tuned **LLaMA 2-7B** chatbot for empathetic and contextual conversations.
- Based on **CBT therapy** prompts with custom logic.

### 📈 Mood Analysis
- Mood score mapping from emojis/emotions.
- **GMM-based classification** of mood levels.
- **ARIMA forecasting** for 7-day mood prediction.
- Linear regression trend detection & visual summaries.

### 📊 Dashboards
- Patient dashboard: mood record, chat history.
- Therapist dashboard: mood overview (7/15/30 days), predictive reports.

### 🔐 Secure User Authentication
- Role-based login (Therapist or Patient)
- Encrypted password storage with `bcrypt`
- Session management using `streamlit.session_state`

---

## 🛠️ Tech Stack

| Layer        | Technology                      |
|--------------|----------------------------------|
| Frontend     | Streamlit (Python)               |
| Backend      | Flask REST API                   |
| Model        | LLaMA 2-7B fine-tuned with QLoRA |
| Database     | SQLite3, CSV                     |
| Libraries    | `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `matplotlib`, `bcrypt`, `streamlit`, `huggingface/transformers` |

---

## 📂 Project Structure

```bash
CureMate/
│
├── main.py                    # Entry point for Streamlit app
├── therapist.py              # Therapist dashboard logic, GMM, ARIMA, Linear Regression logic
├── patient.py               # Patient interface logic
│
├── models/
│   └── llama_finetuned/      # Saved LLaMA weights & tokenizer
│
├── data/
│   ├── users.csv             # User credentials
│   ├── mood.csv              # Mood records
│   ├── chat_logs.csv         # Chat history
│   └── therapistpatient.csv  # Therapist-patient mappings
│
│
└── README.md
