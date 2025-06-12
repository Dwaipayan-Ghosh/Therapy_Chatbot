import streamlit as st
import pandas as pd
import bcrypt
import os
from datetime import datetime, date, timedelta
import numpy as np
from sklearn.mixture import GaussianMixture # For clustering mood categories
from collections import Counter # Still useful for internal checks/counts
import enum
import logging
import altair as alt # For plotting
from scipy.stats import linregress # For linear regression
from statsmodels.tsa.arima.model import ARIMA # For ARIMA model

# ─────────────────────────────
# Logging Configuration
# ─────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ─────────────────────────────
# Config
# ─────────────────────────────
st.set_page_config(page_title="Therapist Dashboard", page_icon="🧠", layout="wide")

# ─────────────────────────────
# Session Defaults
# ─────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.name = ""
    st.session_state.email = ""
    st.session_state.show_add_form = False
    st.session_state.viewing_patient_email = None # State to hold the email of the patient whose report is being viewed
    st.session_state.viewing_patient_name = None # State to hold the name of the patient whose report is being viewed

# ─────────────────────────────
# File Paths
# ─────────────────────────────
THERAPISTS_FILE = "therapists.csv" # Exclusively for therapist login/registration
USERS_FILE = "users.csv" # For patient registration data
THERAPIST_PATIENT_FILE = "therapistPatient.csv" # Stores therapist-patient assignments
MOOD_LOGS_FILE = "mood.csv" # Path to the mood logs CSV file

# ─────────────────────────────
# Enums for User Role (Consistent with users.csv role structure)
# ─────────────────────────────
class UserRole(enum.Enum):
    PATIENT = "patient"
    THERAPIST = "therapist" # Defined here for consistency, but therapist auth uses CSV

# ─────────────────────────────
# File Functions (for therapist management - unchanged from your provided code, uses CSVs)
# ─────────────────────────────
def load_therapists():
    """Loads therapist user data from therapists.csv."""
    try:
        return pd.read_csv(THERAPISTS_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["name", "email", "password"])

def save_therapists(df):
    """Saves therapist user data to therapists.csv."""
    df.to_csv(THERAPISTS_FILE, index=False)

def load_users():
    """Loads users data from users.csv.
    This is used for patient validation during assignment and registration status display.
    """
    try:
        # Ensure 'role' is read as string to prevent issues with mixed types
        return pd.read_csv(USERS_FILE, dtype={"email": str, "name": str, "role": str})
    except FileNotFoundError:
        return pd.DataFrame(columns=["email", "name", "role"]) # Columns as per your users.csv

def load_therapist_patients():
    """Loads therapist-patient assignments from therapistPatient.csv."""
    try:
        return pd.read_csv(THERAPIST_PATIENT_FILE)
    except FileNotFoundError:
        return pd.DataFrame(columns=["therapist_name", "therapist_email", "patient_name", "patient_email"])

def save_therapist_patients(df):
    """Saves therapist-patient assignments to therapistPatient.csv."""
    df.to_csv(THERAPIST_PATIENT_FILE, index=False)

# ─────────────────────────────
# Mood Data Loading Function (from CSV - fetching and filtering by email)
# ─────────────────────────────
def get_mood_entries_by_email_from_csv(user_email: str, days: int = 365) -> pd.DataFrame:
    """
    Retrieves mood entries for a given user email from mood.csv within a specified number of days.
    Filters by email (acting as primary key) and date range.
    Returns a Pandas DataFrame.
    """
    # Define the columns and their desired dtypes for an empty DataFrame
    columns_and_dtypes = {
        "username": str,
        "email": str,
        "datetime": 'datetime64[ns]', # Explicitly set for datetime
        "mood": str,
        "score": float
    }

    if not os.path.exists(MOOD_LOGS_FILE) or os.path.getsize(MOOD_LOGS_FILE) == 0:
        logging.info(f"'{MOOD_LOGS_FILE}' not found or is empty. Returning empty DataFrame.")
        # Return an empty DataFrame with correct dtypes
        return pd.DataFrame(columns=columns_and_dtypes.keys()).astype(columns_and_dtypes)

    try:
        df = pd.read_csv(MOOD_LOGS_FILE)
        # Explicitly convert 'datetime' column to datetime objects
        # 'errors='coerce'' will turn unparseable dates into NaT (Not a Time)
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        
        # Filter out rows where datetime conversion failed (NaT)
        df = df.dropna(subset=['datetime'])

        # Filter by email (email as primary key for user's mood data)
        df_filtered_by_email = df[df['email'] == user_email]

        # Filter by date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        df_filtered_by_date = df_filtered_by_email[
            (df_filtered_by_email['datetime'] >= start_date) &
            (df_filtered_by_email['datetime'] <= end_date)
        ]
        
        return df_filtered_by_date.sort_values(by='datetime').reset_index(drop=True)
    except pd.errors.EmptyDataError:
        logging.info(f"'{MOOD_LOGS_FILE}' is empty after all. Returning empty DataFrame.")
        return pd.DataFrame(columns=columns_and_dtypes.keys()).astype(columns_and_dtypes)
    except Exception as e:
        logging.error(f"Error fetching mood entries for {user_email} from {MOOD_LOGS_FILE}: {e}")
        return pd.DataFrame(columns=columns_and_dtypes.keys()).astype(columns_and_dtypes)


# ─────────────────────────────
# Auth Functions (for therapist login - ONLY therapists.csv)
# ─────────────────────────────
def signup_therapist(name, email, password):
    """
    Registers a new therapist by saving their credentials to therapists.csv.
    This does NOT interact with users.csv.
    """
    df = load_therapists()
    if email in df["email"].values:
        return False
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    new_row = pd.DataFrame([[name, email, hashed_pw]], columns=["name", "email", "password"])
    df = pd.concat([df, new_row], ignore_index=True)
    save_therapists(df)
    return True

def authenticate_therapist(email, password):
    """
    Authenticates a therapist by checking credentials against therapists.csv.
    This does NOT interact with users.csv.
    """
    df = load_therapists()
    user = df[df["email"] == email]
    if not user.empty and bcrypt.checkpw(password.encode(), user.iloc[0]["password"].encode()):
        return True, user.iloc[0]["name"]
    return False, ""

# ─────────────────────────────
# Mood Analysis Logic (GMM and Direct Classification)
# ─────────────────────────────

# Emoji picker map (using text descriptions and their associated scores)
EMOJI_MOODS = {
    "Joy": 0.9, "Contentment": 0.8, "Peacefulness": 0.7, "Gratitude": 0.8, "Hope": 0.6,
    "Love": 0.9, "Excitement": 0.7, "Enthusiasm": 0.7, "Awe": 0.6, "Interest": 0.5,
    "Pride": 0.6, "Amusement": 0.7, "Relief": 0.6, "Compassion": 0.7, "Serenity": 0.8,
    "Sadness": -0.6, "Anxiety": -0.7, "Anger": -0.8, "Fear": -0.9, "Guilt": -0.7,
    "Shame": -0.8, "Disgust": -0.7, "Frustration": -0.6, "Disappointment": -0.6,
    "Worry": -0.7, "Stress": -0.8, "Loneliness": -0.7
}

# --- GMM Clustering Setup (defines the 6 fixed mood categories) ---
mood_scores_array = np.array(list(EMOJI_MOODS.values())).reshape(-1, 1)

try:
    logging.info("Initializing Gaussian Mixture Model and fitting 6 components...")
    # Using GaussianMixture instead of KMeans for probabilistic clustering
    # n_components is equivalent to n_clusters (our k=6)
    # covariance_type='full' allows for arbitrary covariance matrices for each component
    # random_state for reproducibility, n_init for multiple initializations
    gmm_model = GaussianMixture(n_components=6, random_state=42, n_init=10, covariance_type='full')
    gmm_model.fit(mood_scores_array)

    # GMM 'means_' are the centers of the Gaussian components, similar to KMeans 'cluster_centers_'
    CLUSTER_CENTROIDS = sorted([m[0] for m in gmm_model.means_]) # Extract mean value from array and sort
    
    # Assign names based on the sorted centroids
    CLUSTER_NAMES = {
        CLUSTER_CENTROIDS[0]: "Extremely Negative",
        CLUSTER_CENTROIDS[1]: "Moderately Negative",
        CLUSTER_CENTROIDS[2]: "Slightly Negative",
        CLUSTER_CENTROIDS[3]: "Slightly Positive",
        CLUSTER_CENTROIDS[4]: "Moderately Positive",
        CLUSTER_CENTROIDS[5]: "Extremely Positive"
    }
    logging.info(f"GMM Component Means (6 clusters): {CLUSTER_CENTROIDS}")
    logging.info(f"Assigned Cluster Names (6 clusters): {CLUSTER_NAMES}")

except Exception as e:
    logging.error(f"Error during GMM clustering initialization: {e}")
    # Fallback centroids and names if GMM fails (shouldn't happen with simple 1D data)
    CLUSTER_CENTROIDS = [-0.9, -0.6, -0.2, 0.2, 0.6, 0.9] # Approximate typical centroids
    CLUSTER_NAMES = {
        CLUSTER_CENTROIDS[0]: "Error: Extremely Negative",
        CLUSTER_CENTROIDS[1]: "Error: Moderately Negative",
        CLUSTER_CENTROIDS[2]: "Error: Slightly Negative",
        CLUSTER_CENTROIDS[3]: "Error: Slightly Positive",
        CLUSTER_CENTROIDS[4]: "Error: Moderately Positive",
        CLUSTER_CENTROIDS[5]: "Error: Extremely Positive"
    }

# This function will now be used for all mood score classifications (individual and averages)
def classify_mood_score_to_category(score: float) -> str:
    """Classifies a single mood score (or an average score) into one of the 6 GMM-defined categories."""
    if not CLUSTER_CENTROIDS: return "Classification Error" # Fallback if centroids are not defined
    
    closest_centroid_value = None
    min_distance = float('inf')
    for centroid_value in CLUSTER_CENTROIDS:
        distance = abs(score - centroid_value)
        if distance < min_distance:
            min_distance = distance
            closest_centroid_value = centroid_value
    
    return CLUSTER_NAMES.get(closest_centroid_value, "Unknown Category")


# ─────────────────────────────
# Mood Report UI Functions (for Therapist view)
# ─────────────────────────────
def render_patient_mood_report(patient_email, patient_name):
    """
    Renders the mood report for a specific patient, including recent logs and aggregated classifications.
    """
    st.markdown(f"## Mood Report for **{patient_name}** ({patient_email})")

    # Fetch all mood entries (up to last year) for calculations (now from CSV)
    all_mood_entries_df = get_mood_entries_by_email_from_csv(patient_email, days=365)

    # --- Section: Mood Overview (Averages & Classifications) ---
    st.markdown("---")
    st.subheader("Mood Overview by Period")
    
    col7, col15, col30 = st.columns(3)

    # Calculate and display metrics for last 7 days
    weekly_df = all_mood_entries_df[all_mood_entries_df['datetime'] >= (datetime.utcnow() - timedelta(days=7))]
    weekly_avg_mood = weekly_df['score'].mean() if not weekly_df.empty else 0.0
    # Direct classification of average mood score using GMM-defined categories
    weekly_classification = classify_mood_score_to_category(weekly_avg_mood)

    with col7:
        st.metric("Last 7 Days Mood Score", f"{weekly_avg_mood:.2f}")
        st.markdown(f"**Classification:** `{weekly_classification}`")
        logging.info(f"7-day data points: {len(weekly_df)}. Avg: {weekly_avg_mood:.2f}. Class: {weekly_classification}" if not weekly_df.empty else "7-day data points: 0")


    # Calculate and display metrics for last 15 days
    biweekly_df = all_mood_entries_df[all_mood_entries_df['datetime'] >= (datetime.utcnow() - timedelta(days=15))]
    biweekly_avg_mood = biweekly_df['score'].mean() if not biweekly_df.empty else 0.0
    # Direct classification of average mood score using GMM-defined categories
    biweekly_classification = classify_mood_score_to_category(biweekly_avg_mood)
    with col15:
        st.metric("Last 15 Days Mood Score", f"{biweekly_avg_mood:.2f}")
        st.markdown(f"**Classification:** `{biweekly_classification}`")
        logging.info(f"15-day data points: {len(biweekly_df)}. Avg: {biweekly_avg_mood:.2f}. Class: {biweekly_classification}" if not biweekly_df.empty else "15-day data points: 0")
    
    # Calculate and display metrics for last 30 days
    monthly_df = all_mood_entries_df[all_mood_entries_df['datetime'] >= (datetime.utcnow() - timedelta(days=30))]
    monthly_avg_mood = monthly_df['score'].mean() if not monthly_df.empty else 0.0
    # Direct classification of average mood score using GMM-defined categories
    monthly_classification = classify_mood_score_to_category(monthly_avg_mood)
    with col30:
        st.metric("Last 30 Days Mood Score", f"{monthly_avg_mood:.2f}")
        st.markdown(f"**Classification:** `{monthly_classification}`")
        logging.info(f"30-day data points: {len(monthly_df)}. Avg: {monthly_avg_mood:.2f}. Class: {monthly_classification}" if not monthly_df.empty else "30-day data points: 0")


    # --- Section: Mood Trends Over Last 30 Days ---
    st.markdown("---")
    st.subheader("Mood Trends Over Last 30 Days")

    # Prepare data for plotting and ARIMA
    plot_df = all_mood_entries_df[all_mood_entries_df['datetime'] >= (datetime.utcnow() - timedelta(days=30))].copy()
    
    if not plot_df.empty:
        # Aggregate to daily average mood if multiple entries per day
        daily_avg_mood = plot_df.groupby(plot_df['datetime'].dt.date)['score'].mean().reset_index()
        daily_avg_mood.columns = ['date', 'average_score']
        daily_avg_mood['date'] = pd.to_datetime(daily_avg_mood['date']) # Ensure 'date' is datetime for Altair
        daily_avg_mood = daily_avg_mood.sort_values('date').set_index('date') # Set index for time series ops

        # --- Handle missing dates and interpolate for continuous time series ---
        # Create a full date range for the last 30 days
        full_date_range = pd.date_range(end=datetime.utcnow().date(), periods=30, freq='D')
        full_daily_mood = pd.DataFrame(index=full_date_range)
        full_daily_mood.index.name = 'date'
        
        # Merge existing daily averages with the full date range
        full_daily_mood = full_daily_mood.merge(daily_avg_mood, left_index=True, right_index=True, how='left')
        
        # Interpolate missing values (e.g., if a day has no mood entry)
        # We'll use linear interpolation after filling any leading/trailing NaNs with first/last valid value
        full_daily_mood['average_score_interpolated'] = full_daily_mood['average_score'].fillna(method='ffill').fillna(method='bfill').interpolate(method='linear')

        # Drop original 'average_score' if not needed after interpolation
        full_daily_mood = full_daily_mood.drop(columns=['average_score']).rename(columns={'average_score_interpolated': 'average_score'})

        # Reset index to make 'date' a column again for Altair
        full_daily_mood = full_daily_mood.reset_index()


        # --- Exponential Moving Average (EMA) ---
        # EMA with a span (like window size) of 7 days, giving more weight to recent moods.
        if len(full_daily_mood) > 1:
            full_daily_mood['ema_7_day'] = full_daily_mood['average_score'].ewm(span=7, adjust=False).mean()
        else:
            full_daily_mood['ema_7_day'] = full_daily_mood['average_score'] # If only one point, EMA is just that point

        # --- Linear Regression ---
        trend_summary_text = ""
        forecast_summary_text = ""
        
        if len(full_daily_mood) > 1:
            full_daily_mood['days_since_start'] = (full_daily_mood['date'] - full_daily_mood['date'].min()).dt.days
            slope, intercept, r_value, p_value, std_err = linregress(
                full_daily_mood['days_since_start'], full_daily_mood['average_score']
            )
            full_daily_mood['linear_trend'] = intercept + slope * full_daily_mood['days_since_start']

            # Qualitatively interpret slope
            if slope > 0.06:
                slope_interpretation = "significantly improving"
            elif slope > 0.03:
                slope_interpretation = "moderately improving"
            elif slope > 0.01:
                slope_interpretation = "slightly improving"
            elif slope < -0.06:
                slope_interpretation = "significantly declining"
            elif slope < -0.03:
                slope_interpretation = "moderately declining"
            elif slope < -0.01:
                slope_interpretation = "slightly declining"
            else:
                slope_interpretation = "very stable"

            # Qualitatively interpret R-squared
            r_squared_val = r_value**2
            if r_squared_val >= 0.85:
                r_squared_interpretation = "very consistent and highly predictable"
            elif r_squared_val >= 0.6:
                r_squared_interpretation = "quite consistent and reasonably predictable"
            elif r_squared_val >= 0.3:
                r_squared_interpretation = "somewhat consistent, but still notable variability"
            else:
                r_squared_interpretation = "highly variable and less predictable"
            
            st.markdown(f"**Overall 30-Day Trend (Linear Regression):**")
            st.markdown(f"- The patient's mood has been **{slope_interpretation}**.")
            st.markdown(f"- The trend is **{r_squared_interpretation}**.")

        else:
            slope, intercept, r_value, p_value, std_err = 0, 0, 0, 0, 0
            full_daily_mood['linear_trend'] = full_daily_mood['average_score']
            st.info("Not enough data points for a meaningful linear trend analysis.")

        # --- ARIMA Forecast (Demonstration) ---
        forecast_df = pd.DataFrame() # Initialize empty DataFrame for forecast
        if len(full_daily_mood) >= 10: # ARIMA needs a reasonable number of observations
            try:
                # Set 'date' as index for ARIMA
                arima_data = full_daily_mood.set_index('date')['average_score']

                # Fit ARIMA model (p=1, d=1, q=1 is a common starting point)
                model = ARIMA(arima_data, order=(1,1,1)) 
                model_fit = model.fit()

                # Forecast for the next 7 days
                forecast_steps = 7
                forecast_result = model_fit.predict(start=len(arima_data), end=len(arima_data) + forecast_steps - 1)
                forecast_index = pd.date_range(start=full_daily_mood['date'].max() + timedelta(days=1), periods=forecast_steps, freq='D')
                
                # Create a DataFrame for forecast
                forecast_df = pd.DataFrame({'date': forecast_index, 'arima_forecast': forecast_result})
                forecast_df['date'] = pd.to_datetime(forecast_df['date']) # Ensure datetime for Altair
                
                # Qualitatively interpret ARIMA forecast
                if not forecast_df.empty:
                    start_forecast = forecast_df['arima_forecast'].iloc[0]
                    end_forecast = forecast_df['arima_forecast'].iloc[-1]
                    
                    # Use a small threshold to determine "stable"
                    if abs(end_forecast - start_forecast) < 0.05: # if change is less than 0.05
                        forecast_direction = "remain relatively stable"
                    elif end_forecast > start_forecast:
                        forecast_direction = "continue to improve"
                    else:
                        forecast_direction = "continue to decline"
                    
                    forecast_summary_text = (
                        f"The ARIMA forecast suggests the patient's mood will **{forecast_direction}** over the next {forecast_steps} days. "
                        f"Expected mood score range: `{forecast_df['arima_forecast'].min():.2f}` to `{forecast_df['arima_forecast'].max():.2f}`."
                    )
                    st.markdown(f"**ARIMA Forecast for next {forecast_steps} days:**")
                    st.dataframe(forecast_df.set_index('date').rename(columns={'arima_forecast': 'Forecasted Mood Score'}), use_container_width=True)
                    st.markdown(forecast_summary_text)


            except Exception as e:
                st.warning(f"Could not generate ARIMA forecast (requires sufficient data & stability): {e}")
                logging.error(f"ARIMA Error: {e}")
        else:
            st.info("Not enough data points (need at least 10) to generate an ARIMA forecast.")

        # --- Combine data for plotting ---
        plot_data_with_forecast = full_daily_mood # Start with full_daily_mood
        if not forecast_df.empty:
            # Merge forecast_df into plot_data_with_forecast, aligning by 'date'
            plot_data_with_forecast = pd.concat([full_daily_mood, forecast_df], ignore_index=True)
            # Replace NaNs that might arise from concat if columns don't align perfectly
            plot_data_with_forecast['arima_forecast'] = plot_data_with_forecast['arima_forecast'].replace({0: np.nan}) 

        # --- Plotting with Altair ---
        base = alt.Chart(plot_data_with_forecast).encode(
            x=alt.X('date', title='Date'),
            y=alt.Y('average_score', title='Average Mood Score', scale=alt.Scale(domain=[-1, 1])) # Scale from -1 to 1
        )

        # Raw Data Points (using the interpolated data for consistency)
        points = base.mark_circle(size=60, opacity=0.8).encode(
            tooltip=[
                alt.Tooltip('date', format='%Y-%m-%d', title='Date'),
                alt.Tooltip('average_score', format='.2f', title='Avg. Score')
            ]
        )

        # EMA Line
        ema_line = base.mark_line(color='green', strokeWidth=3).encode(
            y=alt.Y('ema_7_day', title='Exponential Moving Average'),
            tooltip=[
                alt.Tooltip('date', format='%Y-%m-%d', title='Date'),
                alt.Tooltip('ema_7_day', format='.2f', title='EMA Score')
            ]
        )

        # Linear Trend Line
        linear_line = base.mark_line(color='red', strokeWidth=2, strokeDash=[5, 5]).encode(
            y=alt.Y('linear_trend', title='Linear Trend'),
            tooltip=[
                alt.Tooltip('date', format='%Y-%m-%d', title='Date'),
                alt.Tooltip('linear_trend', format='.2f', title='Trend Score')
            ]
        )

        # ARIMA Forecast Line (only if generated)
        if 'arima_forecast' in plot_data_with_forecast.columns and not plot_data_with_forecast['arima_forecast'].isnull().all():
            # Only plot ARIMA forecast points where they are not NaN
            arima_line = alt.Chart(plot_data_with_forecast.dropna(subset=['arima_forecast'])).mark_line(color='purple', strokeWidth=2, strokeDash=[2,2]).encode(
                x=alt.X('date', title='Date'),
                y=alt.Y('arima_forecast', title='ARIMA Forecast'),
                tooltip=[
                    alt.Tooltip('date', format='%Y-%m-%d', title='Date'),
                    alt.Tooltip('arima_forecast', format='.2f', title='Forecast Score')
                ]
            )
            chart = (points + ema_line + linear_line + arima_line).properties(
                title='Patient Mood Trend Over Last 30 Days with Forecast'
            ).interactive()
        else:
            chart = (points + ema_line + linear_line).properties(
                title='Patient Mood Trend Over Last 30 Days'
            ).interactive() # Make chart interactive for zoom/pan


        st.altair_chart(chart, use_container_width=True)
        
        # Overall Summary Statement
        st.markdown("---")
        st.subheader("Overall Mood Trend Summary")
        overall_summary = ""
        
        # Base on linear regression slope interpretation
        if slope_interpretation == "significantly improving" or slope_interpretation == "moderately improving":
            overall_summary += "The patient's long-term mood trend is showing **significant improvement**."
        elif slope_interpretation == "slightly improving":
            overall_summary += "The patient's long-term mood trend is showing **slight improvement**."
        elif slope_interpretation == "significantly declining" or slope_interpretation == "moderately declining":
            overall_summary += "The patient's long-term mood trend is showing a **significant decline**."
        elif slope_interpretation == "slightly declining":
            overall_summary += "The patient's long-term mood trend is showing a **slight decline**."
        else:
            overall_summary += "The patient's long-term mood trend appears **stable**."
        
        if forecast_summary_text:
            overall_summary += f" Based on the ARIMA forecast, their mood is expected to **{forecast_direction}** in the coming week."
        else:
            overall_summary += " More data is needed for a reliable future forecast."

        st.info(overall_summary)

    else:
        st.info("Not enough mood data recorded in the last 30 days to generate trend analysis or forecasts. Please record more moods spanning several days/weeks.")

    if all_mood_entries_df.empty:
        st.info("No mood data yet for this patient. Ask them to record their moods!")


    st.markdown("---")
    if st.button("⬅️ Back to Patient List", key="back_from_report"):
        st.session_state.viewing_patient_email = None
        st.session_state.viewing_patient_name = None
        st.rerun()


# ─────────────────────────────
# Main Navigation and Dashboard Logic (for Therapist)
# ─────────────────────────────
if st.session_state.logged_in:
    # If a patient is selected, render their mood report
    if st.session_state.viewing_patient_email:
        render_patient_mood_report(
            st.session_state.viewing_patient_email,
            st.session_state.viewing_patient_name
        )
    else: # Otherwise, show the main therapist dashboard
        st.title("🧑‍⚕️ Therapist Dashboard")
        st.markdown(f"Welcome, **{st.session_state.name}**")

        if st.button("Logout"):
            for k in ["logged_in", "name", "email", "show_add_form", "viewing_patient_email", "viewing_patient_name"]:
                st.session_state[k] = False if k == "logged_in" else None
            st.rerun()

        # ─────────────────────────────
        # List Patients Assigned
        # ─────────────────────────────
        st.subheader("Your Assigned Patients")

        therapist_email = st.session_state.email
        therapist_patients_df = load_therapist_patients()
        your_assigned_patients = therapist_patients_df[therapist_patients_df["therapist_email"] == therapist_email]

        # Patient registration status now checked against users.csv
        all_registered_patients_df_csv = load_users() # Load from users.csv
        if all_registered_patients_df_csv.empty:
            st.warning("No patients registered in the main app yet. Ask them to sign up via the patient app.")

        if your_assigned_patients.empty:
            st.info("No patients assigned to you yet.")
        else:
            for idx, row in your_assigned_patients.iterrows():
                # Check patient existence in users.csv for status display
                matched_patient_in_csv = all_registered_patients_df_csv[
                    (all_registered_patients_df_csv['email'] == row['patient_email'])
                ]

                with st.container():
                    # Use unique keys for buttons
                    cols = st.columns([3, 3, 2, 1])
                    with cols[0]:
                        # Make the email clickable to view report
                        if st.button(f"📧 **{row['patient_email']}**", key=f"view_report_{row['patient_email']}", use_container_width=True):
                            st.session_state.viewing_patient_email = row['patient_email']
                            st.session_state.viewing_patient_name = row['patient_name']
                            st.rerun()
                    with cols[1]:
                        st.markdown(f"🧑‍💼 **{row['patient_name']}**")
                    with cols[2]:
                        # Display status based on users.csv check
                        if not matched_patient_in_csv.empty:
                            st.success("✅ Registered in Patient App")
                        else:
                            st.error("⚠️ Not registered in Patient App") # Warn if patient email not found in users.csv
                    with cols[3]:
                        if st.button("🗑", key=f"delete_{row['patient_email']}"): # Unique key for delete button
                            therapist_patients_df = therapist_patients_df.drop(index=idx).reset_index(drop=True)
                            save_therapist_patients(therapist_patients_df)
                            st.success("Patient removed.")
                            st.rerun()
                st.markdown("---")

        # ─────────────────────────────
        # Add Patient Section (Retained exact previous logic as requested)
        # ─────────────────────────────
        st.subheader("Assign New Patient")
        if st.button("➕ Assign Patient"):
            st.session_state.show_add_form = True

        if st.session_state.show_add_form:
            with st.expander("Enter patient details", expanded=True):
                with st.form("add_patient_form"):
                    p_email = st.text_input("Patient Email", key="new_patient_email_input")
                    p_name = st.text_input("Patient Name", key="new_patient_name_input")
                    submitted = st.form_submit_button("Assign Patient")

                    if submitted:
                        # --- THIS IS THE LOGIC YOU ASKED TO RETAIN EXACTLY ---
                        # It uses users_df (loaded from users.csv) for validation.
                        users_df = load_users() # Reload to get latest from users.csv
                        valid_patient = users_df[
                            (users_df["email"] == p_email) &
                            (users_df["name"] == p_name) &
                            (users_df["role"].str.lower() == UserRole.PATIENT.value) # Using UserRole Enum
                        ]
                        # ---------------------------------------------------

                        already_added = (
                            (therapist_patients_df["therapist_email"] == st.session_state.email) &
                            (therapist_patients_df["patient_email"] == p_email)
                        ).any()

                        if not (p_email and p_name): # Added check for empty inputs
                            st.error("Please provide both patient email and name.")
                        elif valid_patient.empty: # Uses users.csv validation
                            st.error("❌ Patient not registered or name/role mismatch in users.csv.")
                        elif already_added:
                            st.warning("⚠ Patient already assigned to you.")
                        else:
                            new_row = pd.DataFrame([{
                                "therapist_name": st.session_state.name,
                                "therapist_email": st.session_state.email,
                                "patient_name": p_name,
                                "patient_email": p_email
                            }])
                            therapist_patients_df = pd.concat([therapist_patients_df, new_row], ignore_index=True)
                            save_therapist_patients(therapist_patients_df)
                            st.success("✅ Patient assigned successfully!")
                            st.session_state.show_add_form = False
                            st.rerun()

else: # Login / Register for Therapist
    st.sidebar.title("Therapist Portal")
    nav = st.sidebar.radio("Menu", ["Login", "Register"])

    if nav == "Register":
        st.title("Register as Therapist")
        with st.form("register_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Register")

            if submitted:
                if not (name and email and password):
                    st.warning("Fill all fields.")
                else:
                    # Therapist registration: ONLY to therapists.csv (as per your explicit instruction)
                    if signup_therapist(name, email, password): 
                        st.success("Therapist account created! Please log in.")
                        st.rerun()
                    else:
                        st.error("Email already registered as a therapist.")

    elif nav == "Login":
        st.title("Therapist Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if email and password:
                # Authenticate against local therapists.csv (as per your explicit instruction)
                ok, name = authenticate_therapist(email, password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.name = name
                    st.session_state.email = email
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
            else:
                st.warning("Please enter both email and password.")

