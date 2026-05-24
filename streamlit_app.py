# Imports

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib


# Page Config

st.set_page_config(
    page_title="NZ Traffic Crash Severity Predictor",
    page_icon="🚗",
    layout="wide"
)


# Load Model + Dataset

@st.cache_resource
def load_model():

    # Download model files from Google Drive
    gdown.download(
        "https://drive.google.com/uc?id=1mKPIErvTlYCc_m61ACv8l-gurNhwrKb9",
        "rf_model.pkl",
        quiet=False
    )

    gdown.download(
        "https://drive.google.com/uc?id=1StLwxf24kYBCN8e-8mdABziUewGoss_P",
        "model_columns.pkl",
        quiet=False
    )

    model = joblib.load("rf_model.pkl")
    columns = joblib.load("model_columns.pkl")

    return model, columns


@st.cache_data
def load_data():

    cas = pd.read_csv(
        "https://opendata-nzta.opendata.arcgis.com/datasets/NZTA::crash-analysis-system-cas-data-1.csv"
    )

    return cas


rf_model, model_columns = load_model()
cas = load_data()


# Title + Introduction

st.title("🚗 New Zealand Traffic Crash Severity Predictor")

st.markdown("""
This application predicts the likelihood of a crash resulting in a serious or fatal outcome
using a Random Forest machine learning model trained on the New Zealand
Crash Analysis System (CAS) dataset.
""")

st.divider()


# Sidebar Inputs (Rensei)

st.sidebar.header("Enter Crash Conditions")

speed = st.sidebar.slider(
    "Speed Limit (km/h)",
    0,
    110,
    50
)

weather = st.sidebar.selectbox(
    "Weather Condition",
    [
        "Fine",
        "Light rain",
        "Heavy rain",
        "Mist or Fog",
        "Snow",
        "Hail or Sleet"
    ]
)

road_surface = st.sidebar.selectbox(
    "Road Surface",
    [
        "Sealed",
        "Unsealed",
        "End of seal"
    ]
)

light = st.sidebar.selectbox(
    "Light Condition",
    [
        "Overcast",
        "Bright sun",
        "Dark",
        "Twilight"
    ]
)

region = st.sidebar.selectbox(
    "Region",
    [
        "Auckland Region",
        "Bay of Plenty Region",
        "Canterbury Region",
        "Gisborne Region",
        "Hawke's Bay Region",
        "Manawatū-Whanganui Region",
        "Marlborough Region",
        "Nelson Region",
        "Northland Region",
        "Otago Region",
        "Southland Region",
        "Taranaki Region",
        "Tasman Region",
        "Waikato Region",
        "Wellington Region",
        "West Coast Region"
    ]
)

traffic = st.sidebar.selectbox(
    "Traffic Control",
    [
        "Nil",
        "Stop",
        "Give way",
        "Traffic Signals",
        "School Patrol/warden",
        "Pointsman",
        "Isolated Pedestrian signal (non-intersection)"
    ]
)

urban = st.sidebar.selectbox(
    "Urban / Rural",
    [
        "Urban",
        "Open"
    ]
)


# Prediction Section (Anika)

st.header("🔮 Crash Severity Prediction")

st.write("""
Enter the crash conditions using the sidebar and click the button below
to predict the likelihood of a serious or fatal crash.
""")

if st.button("Predict Crash Severity"):

    # Speed grouping
    if speed <= 50:
        speed_group = "0-50"

    elif speed <= 80:
        speed_group = "50-80"

    else:
        speed_group = "100+"

    # Create empty encoded row
    row = pd.DataFrame([{col: 0 for col in model_columns}])

    # Feature mappings
    mappings = [
        f"weatherA_{weather}",
        f"roadSurface_{road_surface}",
        f"light_{light}",
        f"region_{region}",
        f"trafficControl_{traffic}",
        f"urban_{urban}",
        f"speed_group_{speed_group}",
    ]

    # Activate selected features
    for col in mappings:

        if col in row.columns:
            row[col] = 1

    # Predict
    prediction = rf_model.predict(row)[0]

    proba = rf_model.predict_proba(row)[0]

    high_severity_pct = round(proba[1] * 100, 1)
    low_severity_pct = round(proba[0] * 100, 1)

    # Display results
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Serious/Fatal Crash")

    else:
        st.success("✅ Lower Risk Crash")

    # Probability metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "High Severity Risk",
            f"{high_severity_pct}%"
        )

    with col2:
        st.metric(
            "Low Severity Risk",
            f"{low_severity_pct}%"
        )

st.divider()


# App Graphs / Visualisations (Adem)

st.header("📊 Crash Insights")

chart = st.selectbox(
    "Select a graph",
    [
        "Crash severity distribution",
        "Crashes by year",
        "Top 10 regions by total crashes",
        "Crash severity by region",
        "Crash severity by weather condition"
    ]
)
