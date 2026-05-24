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

    model = joblib.load("random_forest_model.pkl")
    columns = joblib.load("model_columns.pkl")

    return model, columns


@st.cache_data
def load_data():

    cas = pd.read_csv("cas_small.csv")

    return cas


random_forest_model, model_columns = load_model()
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
    prediction = random_forest_model.predict(row)[0]

    proba = random_forest_model.predict_proba(row)[0]

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


# Crash Severity Distribution

if chart == "Crash severity distribution":

    severity_counts = cas["crashSeverity"].value_counts().reindex(
        [
            "Non-Injury Crash",
            "Minor Crash",
            "Serious Crash",
            "Fatal Crash"
        ]
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        severity_counts.index,
        severity_counts.values
    )

    ax.set_title("Crash Severity Distribution")
    ax.set_xlabel("Crash Severity")
    ax.set_ylabel("Number of Crashes")

    st.pyplot(fig)


# Crashes by Year

elif chart == "Crashes by year":

    year_counts = cas["crashYear"].dropna().astype(int)

    year_counts = (
        year_counts[year_counts < 2026]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(
        year_counts.index,
        year_counts.values
    )

    ax.set_title("Total Crashes by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Crashes")

    plt.xticks(rotation=45)

    st.pyplot(fig)


# Top Regions

elif chart == "Top 10 regions by total crashes":

    region_counts = cas["region"].fillna("Unknown")

    region_counts = region_counts.str.replace(
        " Region",
        "",
        regex=False
    )

    top_regions = (
        region_counts.value_counts()
        .head(10)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        top_regions.index,
        top_regions.values
    )

    ax.set_title("Top 10 Regions by Total Crashes")
    ax.set_xlabel("Number of Crashes")
    ax.set_ylabel("Region")

    st.pyplot(fig)


# Crash Severity by Region

elif chart == "Crash severity by region":

    region_data = cas[["region", "crashSeverity"]].copy()

    region_data["region"] = (
        region_data["region"]
        .fillna("Unknown")
        .str.replace(" Region", "", regex=False)
    )

    region_severity = pd.crosstab(
        region_data["region"],
        region_data["crashSeverity"]
    )

    region_severity["total"] = region_severity.sum(axis=1)

    region_severity = (
        region_severity
        .sort_values("total", ascending=False)
        .head(10)
        .drop("total", axis=1)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    region_severity.plot(
        kind="barh",
        stacked=True,
        ax=ax
    )

    ax.set_title("Crash Severity by Region")
    ax.set_xlabel("Number of Crashes")
    ax.set_ylabel("Region")

    st.pyplot(fig)


# Crash Severity by Weather

elif chart == "Crash severity by weather condition":

    weather_data = cas[
        ["weatherA", "crashSeverity"]
    ].copy()

    weather_data["weatherA"] = (
        weather_data["weatherA"]
        .fillna("Unknown")
    )

    top_weather = (
        weather_data["weatherA"]
        .value_counts()
        .head(10)
        .index
    )

    weather_severity = pd.crosstab(
        weather_data["weatherA"],
        weather_data["crashSeverity"]
    )

    weather_severity = weather_severity.loc[top_weather]

    fig, ax = plt.subplots(figsize=(10, 6))

    weather_severity.plot(
        kind="bar",
        stacked=True,
        ax=ax
    )

    ax.set_title("Crash Severity by Weather Condition")
    ax.set_xlabel("Weather Condition")
    ax.set_ylabel("Number of Crashes")

    plt.xticks(rotation=0)

    st.pyplot(fig)

st.divider()


# Model Performance (Your Section)

st.header("📈 Model Performance")

performance = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        0.6255,
        0.6588,
        0.6901
    ],
    "F1 Score": [
        0.1750,
        0.1763,
        0.1862
    ]
})

st.dataframe(performance)

st.markdown("""
Random Forest achieved the strongest overall performance across the evaluated
machine learning models and was therefore selected for deployment in the final application.
""")

st.divider()


# About Section

st.header("ℹ️ About This Project")

st.markdown("""
This project was developed using the New Zealand Crash Analysis System (CAS) dataset.

The goal of the project was to:
- explore crash trends and conditions,
- analyse crash severity patterns,
- train machine learning models,
- and develop an interactive web application capable of predicting crash severity risk.

The final deployed model uses Random Forest classification to estimate the
likelihood of a serious or fatal crash based on selected crash conditions.
""")
