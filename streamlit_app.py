import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================================================
# Load Model Files
# ==========================================================

model = joblib.load("random_forest_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ==========================================================
# Page Setup
# ==========================================================

st.set_page_config(
    page_title="NZ Crash Severity Predictor",
    page_icon="🚗",
    layout="wide"
)

# ==========================================================
# Title
# ==========================================================

st.title("🚗 NZ Crash Severity Predictor")

st.markdown("""
Predict whether crash conditions are likely to result in a
serious or fatal crash using a Random Forest machine learning model.
""")

# ==========================================================
# Sidebar Inputs
# ==========================================================

st.sidebar.header("Crash Conditions")

weatherA = st.sidebar.selectbox(
    "Weather Condition",
    [
        "Fine",
        "Light rain",
        "Heavy rain",
        "Mist or fog",
        "Snow",
        "Hail or sleet"
    ]
)

roadSurface = st.sidebar.selectbox(
    "Road Surface",
    [
        "Sealed",
        "Unsealed",
        "End of sealed road"
    ]
)

light = st.sidebar.selectbox(
    "Light Condition",
    [
        "Bright sun",
        "Overcast",
        "Dark",
        "Twilight"
    ]
)

urban = st.sidebar.selectbox(
    "Area Type",
    [
        "Urban",
        "Open"
    ]
)

region = st.sidebar.selectbox(
    "Region",
    [
        "Auckland Region",
        "Canterbury Region",
        "Wellington Region",
        "Waikato Region",
        "Otago Region"
    ]
)

speedLimit = st.sidebar.slider(
    "Speed Limit",
    min_value=0,
    max_value=120,
    value=50
)

# ==========================================================
# Create Input Data
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
# Encode Input
# ==========================================================

input_encoded = pd.get_dummies(input_data)

input_encoded = input_encoded.reindex(
    columns=model_columns,
    fill_value=0
)

# ==========================================================
# Prediction
# ==========================================================

if st.button("Predict Crash Severity"):

    prediction = model.predict(input_encoded)[0]

    probability = model.predict_proba(input_encoded)[0][1]

    st.header("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Severity Risk")
    else:
        st.success("✅ Low Severity Risk")

    st.metric(
        "Probability of Serious/Fatal Crash",
        f"{probability:.1%}"
    )

# ==========================================================
# Model Performance
# ==========================================================

st.header("📊 Model Performance")

performance = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [0.6255, 0.6588, 0.6901],
    "F1 Score": [0.1750, 0.1763, 0.1862]
})

st.dataframe(performance)

# ==========================================================
# F1 Score Chart
# ==========================================================

fig, ax = plt.subplots()

ax.bar(
    performance["Model"],
    performance["F1 Score"]
)

ax.set_ylabel("F1 Score")
ax.set_title("Model Comparison")

st.pyplot(fig)

# ==========================================================
# About Section
# ==========================================================

st.header("ℹ️ About")

st.markdown("""
This application was developed using the New Zealand Crash Analysis System (CAS) dataset.

Random Forest was selected as the final deployed model after outperforming:
- Logistic Regression
- Decision Tree

The app predicts whether crash conditions are associated with a higher likelihood
of serious or fatal crashes.
""")
