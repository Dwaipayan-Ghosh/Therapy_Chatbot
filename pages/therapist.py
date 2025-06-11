import streamlit as st
import pandas as pd
import bcrypt
import os

# ───────────────────────────
# Page config & session init
# ───────────────────────────
st.set_page_config(page_title="Therapist Dashboard", page_icon="🔧")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.name = ""
    st.session_state.email = ""
    st.session_state.show_add_form = False

# ───────────────────────────
# File paths
# ───────────────────────────
THERAPISTS_FILE = "therapists.csv"
PATIENTS_FILE = "patients.csv"
USERS_FILE = "users.csv"

# ───────────────────────────
# Auth helpers
# ───────────────────────────
def load_therapists():
    try:
        return pd.read_csv(THERAPISTS_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["name", "email", "password"])

def save_therapists(df):
    df.to_csv(THERAPISTS_FILE, index=False)

def signup_therapist(name, email, pw):
    df = load_therapists()
    if email in df["email"].values:
        return False
    hashed = bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()
    df = pd.concat([df, pd.DataFrame([[name, email, hashed]], columns=["name", "email", "password"])])
    save_therapists(df)
    return True

def authenticate_therapist(email, pw):
    df = load_therapists()
    user = df[df["email"] == email]
    if not user.empty and bcrypt.checkpw(pw.encode(), user.iloc[0]["password"].encode()):
        return True, user.iloc[0]["name"]
    return False, ""

# ───────────────────────────
# Patient helpers
# ───────────────────────────
def load_patients():
    if os.path.exists(PATIENTS_FILE):
        return pd.read_csv(PATIENTS_FILE)
    return pd.DataFrame(columns=["email", "name", "profile"])

def save_patients(df):
    df.to_csv(PATIENTS_FILE, index=False)

def load_users():
    if os.path.exists(USERS_FILE):
        return pd.read_csv(USERS_FILE)
    return pd.DataFrame(columns=["email", "name", "role"])

# ───────────────────────────
# Navigation
# ───────────────────────────
if st.session_state.logged_in:
    nav = "Dashboard"
else:
    nav = st.sidebar.radio("Therapist Portal", ["Login", "Register"])

# ───────────────────────────
# Register
# ───────────────────────────
if nav == "Register" and not st.session_state.logged_in:
    st.title("Create a Therapist Account")
    with st.form("register"):
        name = st.text_input("Full name")
        email = st.text_input("E-mail")
        pw = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign up")

        if ok:
            if not (name and email and pw):
                st.warning("Please fill every field.")
            elif signup_therapist(name, email, pw):
                st.success("✅ Account created! Please log in.")
                st.rerun()
            else:
                st.error("That e-mail is already registered.")

# ───────────────────────────
# Login
# ───────────────────────────
elif nav == "Login" and not st.session_state.logged_in:
    st.title("Therapist Login")
    email = st.text_input("E-mail")
    pw = st.text_input("Password", type="password")
    if st.button("Log in"):
        if not (email and pw):
            st.warning("Enter both e-mail and password.")
        else:
            ok, name = authenticate_therapist(email, pw)
            if ok:
                st.session_state.logged_in = True
                st.session_state.name = name
                st.session_state.email = email
                st.rerun()
            else:
                st.error("Invalid credentials.")

# ───────────────────────────
# Dashboard
# ───────────────────────────
elif st.session_state.logged_in:
    st.title("🧑‍⚕️ Therapist Dashboard")
    st.markdown(f"Welcome, **{st.session_state.name}**!")

    if st.button("Log out"):
        for k in ["logged_in", "name", "email", "show_add_form"]:
            st.session_state[k] = False if k == "logged_in" else ""
        st.rerun()

    patients_df = load_patients()
    users_df = load_users()

    # ───────────────────────────
    # Patient List
    # ───────────────────────────
    st.subheader("Patient List")

    if patients_df.empty:
        st.info("No patients added yet.")
    else:
        for idx, row in patients_df.iterrows():
            with st.container():
                cols = st.columns([3, 3, 1])
                with cols[0]:
                    if st.button(row["email"], key=f"email_{idx}"):
                        st.session_state["selected_patient_email"] = row["email"]
                        st.switch_page("moodReport.py")

                with cols[1]:
                    st.markdown(
                        f"<div style='padding: 0.5rem; background-color: #1e1e1e; border-radius: 10px; text-align: center; font-weight: bold;'>{row['name']}</div>",
                        unsafe_allow_html=True
                    )

                with cols[2]:
                    if st.button("🗑️", key=f"delete_{idx}"):
                        patients_df = patients_df.drop(idx).reset_index(drop=True)
                        save_patients(patients_df)
                        st.success(f"Deleted patient: {row['name']}")
                        st.rerun()

                st.markdown("<hr style='border: 1px solid #444;' />", unsafe_allow_html=True)

    # ───────────────────────────
    # Add Patient
    # ───────────────────────────
    st.subheader("Add Patient")
    if st.button("➕ Add Patient"):
        st.session_state.show_add_form = True

    if st.session_state.get("show_add_form"):
        with st.expander("Enter patient details", expanded=True):
            with st.form("add_patient"):
                p_email = st.text_input("Patient e-mail")
                p_name = st.text_input("Patient name")
                submitted = st.form_submit_button("Submit")

                if submitted:
                    if not (p_email and p_name):
                        st.error("❌ Fill out all fields.")
                    else:
                        patients_df = load_patients()
                        users_df = load_users()

                        already_in_list = ((patients_df["email"] == p_email) & (patients_df["name"] == p_name)).any()
                        registered_user = (
                            (users_df["email"] == p_email) &
                            (users_df["name"] == p_name) &
                            (users_df["role"].str.lower() == "patient")
                        ).any()

                        if not registered_user:
                            st.error("❌ This user is not registered as a patient.")
                        elif already_in_list:
                            st.warning("⚠️ Patient already exists in your list.")
                        else:
                            new_row = pd.DataFrame([[p_email, p_name, None]], columns=["email", "name", "profile"])
                            patients_df = pd.concat([patients_df, new_row], ignore_index=True)
                            save_patients(patients_df)
                            st.success("✅ Patient added successfully!")
                            st.session_state.show_add_form = False
                            st.rerun()
