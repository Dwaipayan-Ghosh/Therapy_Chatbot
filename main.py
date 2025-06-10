import streamlit as st

# Hide the sidebar navigation using layout and menu config
st.set_page_config(
    page_title="Therapy Chatbot",
    page_icon="💬",
    layout="centered",  # Optional: can be 'wide' or 'centered'
    initial_sidebar_state="collapsed",
    menu_items={
        'Get help': None,
        'Report a bug': None,
        'About': None
    }
)

# Hide sidebar pages completely with custom CSS
hide_pages_style = """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
"""
st.markdown(hide_pages_style, unsafe_allow_html=True)

# Your existing UI
st.title("🧠 Who are you?")
st.markdown("Please select your role to continue:")

role = st.radio("Choose your role:", ["Patient", "Therapist"], horizontal=True)

if st.button("Continue"):
    if role == "Patient":
        st.switch_page("pages/patient.py")
    else:
        st.switch_page("pages/therapist.py")
