import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================================================
# Load Dataset + Model
# ==========================================================

# Load CAS dataset
cas = pd.read_csv(
    "https://opendata-nzta.opendata.arcgis.com/datasets/NZTA::crash-analysis-system-cas-data-1.csv"
)

# Load trained Random Forest model
model = joblib.load("random_forest_model.pkl")

# Load model training columns
model_columns = joblib.load("model_columns.pkl")

# ==========================================================
# Streamlit Page Settings
# ==========================================================

st.set_page_config(
    page_title="NZ Traffic Accident Risk Predictor",
    page_icon="🚗",
    layout="wide"
)

# ==========================================================
# App Title
# ==========================================================

st.title("🚗 New Zealand Traffic Accident Risk Predictor")

st.markdown("""
This application predicts whether crash conditions are likely to result in a
**Serious or Fatal Crash** using a Random Forest machine learning model trained
on the New Zealand Crash Analysis System (CAS) dataset.
""")

# ==========================================================
# Sidebar Inputs
# ==========================================================

st.sidebar.header("Enter Crash Conditions")

# Use REAL dataset categories to prevent mismatches

weather_options = sorted(
    cas['weatherA'].dropna().astype(str).unique()
)

road_surface_options = sorted(
    cas['roadSurface'].dropna().astype(str).unique()
)

light_options = sorted(
    cas['light'].dropna().astype(str).unique()
)

region_options = sorted(
    cas['region'].dropna().astype(str).unique()
)

urban_options = sorted(
    cas['urban'].dropna().astype(str).unique()
)

# ==========================================================
# User Inputs
# ==========================================================

weatherA = st.sidebar.selectbox(
    "Weather Condition",
    weather_options
)

roadSurface = st.sidebar.selectbox(
    "Road Surface",
    road_surface_options
)

light = st.sidebar.selectbox(
    "Light Condition",
    light_options
)

speedLimit = st.sidebar.slider(
    "Speed Limit",
    min_value=0,
    max_value=120,
    value=50
)

urban = st.sidebar.selectbox(
    "Urban Area",
    urban_options
)

region = st.sidebar.selectbox(
    "Region",
    region_options
)

# ==========================================================
# Create Input DataFrame
# ==========================================================

input_data = pd.DataFrame({
    'weatherA': [weatherA],
    'roadSurface': [roadSurface],
    'light': [light],
    'speedLimit': [speedLimit],
    'urban': [urban],
    'region': [region]
})

# ==========================================================
# One-Hot Encode Input
# ==========================================================

input_encoded = pd.get_dummies(input_data)

# Match training columns exactly
input_encoded = input_encoded.reindex(
    columns=model_columns,
    fill_value=0
)

# ==========================================================
# Prediction Section
# ==========================================================

if st.button("Predict Crash Severity"):

    prediction = model.predict(input_encoded)[0]

    probability = model.predict_proba(input_encoded)[0][1]

    st.header("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Severity Risk (Serious or Fatal Crash)")
    else:
        st.success("✅ Low Severity Risk")

    st.metric(
        "Probability of High Severity Crash",
        f"{probability:.1%}"
    )

# ==========================================================
# EDA Visualisations
# ==========================================================

st.header("📊 Crash Insights")

# Crash Severity Counts
severity_counts = cas['crashSeverity'].value_counts()

fig1, ax1 = plt.subplots(figsize=(6, 4))

ax1.bar(
    severity_counts.index,
    severity_counts.values
)

ax1.set_title("Crash Severity Distribution")
ax1.set_xlabel("Crash Severity")
ax1.set_ylabel("Number of Crashes")

st.pyplot(fig1)

# Top Regions
top_regions = cas['region'].value_counts().head(10)

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.barh(
    top_regions.index,
    top_regions.values
)

ax2.set_title("Top 10 Regions by Number of Crashes")
ax2.set_xlabel("Crash Count")

st.pyplot(fig2)

# Weather Conditions
weather_counts = cas['weatherA'].value_counts().head(10)

fig3, ax3 = plt.subplots(figsize=(8, 5))

ax3.bar(
    weather_counts.index,
    weather_counts.values
)

ax3.set_title("Top Weather Conditions")
ax3.set_xlabel("Weather Condition")
ax3.set_ylabel("Crash Count")

plt.xticks(rotation=45)

st.pyplot(fig3)

# ==========================================================
# Model Performance
# ==========================================================

st.header("📈 Model Performance")

performance = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [0.6255, 0.6588, 0.6901],
    "Precision": [0.1025, 0.1051, 0.1129],
    "Recall": [0.5961, 0.5482, 0.5324],
    "F1 Score": [0.1750, 0.1763, 0.1862]
})

st.dataframe(performance)

# ==========================================================
# About Section
# ==========================================================

st.header("ℹ️ About This Project")

st.markdown("""
This project was developed using the New Zealand Crash Analysis System (CAS) dataset.

Three machine learning models were evaluated:
- Logistic Regression
- Decision Tree
- Random Forest

Random Forest achieved the highest F1 Score and was selected as the final model
for deployment in this application.
""")
