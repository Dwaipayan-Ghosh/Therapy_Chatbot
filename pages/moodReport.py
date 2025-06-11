import streamlit as st

st.set_page_config(page_title="Mood Report", page_icon="📈")

st.title("📈 Mood Report")

selected_email = st.session_state.get("selected_patient_email")

if selected_email:
    st.success(f"Viewing mood report for: **{selected_email}**")
    # Here you can add actual mood report charts or data retrieval logic
else:
    st.warning("No patient selected. Please go back to the dashboard.")
