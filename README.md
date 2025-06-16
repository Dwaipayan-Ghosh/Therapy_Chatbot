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
THERAPY_CHATBOT/
├── local_model/ # Fine-tuned LLaMA 2 chatbot logic
│ ├── psychologisv2-8.0B... # GGUF quantized model file
│ └── therapyllama.py # Chatbot response logic
│
├── pages/ # Streamlit multipage views
│ ├── patient.py # Patient dashboard
│ └── therapist.py # Therapist dashboard
│
├── data files
│ ├── users.csv # Registered users (patients & therapists)
│ ├── chat_logs.csv # Stored chatbot interactions
│ ├── mood.csv # Daily mood entries
│ ├── therapistPatient.csv # Therapist–patient mapping
│ ├── therapists.csv # Therapist records
│ └── global.db # SQLite DB (viewed via view_db.py)
│
├── main.py # Main entrypoint for the Streamlit app
├── dev.py # Utility/test script
├── train.py # Training script for mood classification
├── test.ipynb # Notebook for model testing
├── text_classification.py # GMM and ARIMA model logic
├── view_db.py # View SQLite DB content
├── requirements.txt # Python dependencies
└── README.md
---

## 💡 Features

🟢 For Patients:
- Role-based login via Streamlit.
- Submit mood via emoji/emotion dropdown.
- Chat with a therapy-oriented LLaMA 2 chatbot.
- View recent mood summaries and chatbot logs.

🔵 For Therapists:
- View all assigned patients.
- Daily mood trend summary over 7, 15, and 30 days.
- GMM-based emotional cluster analysis.
- ARIMA-based 7-day emotional forecast.

---

## 🤖 AI Chatbot

- Uses LLaMA 2–7B, finetuned on 100 CBT-based instruction-response pairs.
- Model format: GGUF (4-bit quantized for local inference)
- Handled via local_model/therapyllama.py
- Responses are generated using context-aware instruction prompts.

---

## 🔧 Setup Instructions

1. Clone the Repository:

```bash
git clone https://github.com/Dwaipayan-Ghosh/Therapy_Chatbot

2. Create virtual environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

3. Install requirements
```bash
pip install -r requirements.txt

4. Run the app
```bash
streamlit run main.py
