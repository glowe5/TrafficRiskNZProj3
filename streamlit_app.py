import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==========================================================
# Load Model and Feature Columns
# ==========================================================
model = joblib.load("random_forest_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# ==========================================================
# App Configuration
# ==========================================================
st.set_page_config(
    page_title="NZ Traffic Accident Risk Predictor",
    page_icon="🚗",
    layout="wide"
)

# ==========================================================
# Title and Introduction
# ==========================================================
st.title("🚗 New Zealand Traffic Accident Risk Predictor")
st.markdown("""
This application predicts whether a set of crash conditions is likely to
result in a **Serious or Fatal Crash**.

The prediction is based on a **Random Forest** model trained using the
New Zealand Crash Analysis System (CAS) dataset.
""")

# ==========================================================
# Sidebar Inputs
# ==========================================================
st.sidebar.header("Enter Crash Conditions")

weatherA = st.sidebar.selectbox(
    "Weather Condition",
    ["Fine", "Light rain", "Heavy rain", "Mist or Fog", "Snow", "Hail or Sleet"]
)

roadSurface = st.sidebar.selectbox(
    "Road Surface",
    ["Sealed", "Unsealed", "End of Sealed Road"]
)

light = st.sidebar.selectbox(
    "Light Condition",
    ["Dark", "Overcast", "Bright Sun", "Twilight"]
)

speed_group = st.sidebar.selectbox(
    "Speed Group",
    ["0-50", "50-80", "100+"]
)

urban = st.sidebar.selectbox(
    "Area Type",
    ["Urban", "Open"]
)

trafficControl = st.sidebar.selectbox(
    "Traffic Control",
    ["Nil", "Traffic Signals", "Stop Sign", "Give Way Sign", "Unknown"]
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

# ==========================================================
# Prepare Input Data
# ==========================================================
input_data = pd.DataFrame({
    "weatherA": [weatherA],
    "roadSurface": [roadSurface],
    "light": [light],
    "speed_group": [speed_group],
    "urban": [urban],
    "trafficControl": [trafficControl],
    "region": [region]
})

# One-hot encode user input
input_encoded = pd.get_dummies(input_data)

# Match training columns
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
        st.error("⚠️ High Severity Risk (Serious or Fatal Crash)")
    else:
        st.success("✅ Low Severity Risk (Non-Injury or Minor Crash)")

    st.metric(
        "Probability of High Severity Crash",
        f"{probability:.1%}"
    )

    if probability >= 0.50:
        st.warning(
            "These conditions are associated with a higher likelihood "
            "of serious or fatal crashes."
        )
    else:
        st.info(
            "These conditions are associated with a lower likelihood "
            "of serious or fatal crashes."
        )

# ==========================================================
# Data Insights
# ==========================================================
st.header("📊 Key Project Insights")

st.markdown("""
### Main Findings
- **Random Forest** was the best-performing model.
- **Speed limit** was the most important predictor.
- **Urban/rural setting** significantly influenced crash severity.
- **Weather and lighting conditions** also affected outcomes.
- Predicting severe crashes is challenging because they are relatively rare.
""")

# ==========================================================
# Model Performance Table
# ==========================================================
st.subheader("Model Performance Comparison")

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

st.dataframe(performance, use_container_width=True)

# ==========================================================
# F1 Score Chart
# ==========================================================
st.subheader("F1 Score Comparison")

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(performance["Model"], performance["F1 Score"])
ax.set_ylabel("F1 Score")
ax.set_title("F1 Score by Model")
st.pyplot(fig)

# ==========================================================
# About Section
# ==========================================================
st.header("ℹ️ About This Project")

st.markdown("""
This project was developed for a data science assignment using the
**New Zealand Crash Analysis System (CAS)** dataset.

The aim was to predict whether crash conditions would result in a
serious or fatal crash.

Three machine learning models were evaluated:
- Logistic Regression
- Decision Tree
- Random Forest

The **Random Forest** model achieved the highest F1 Score and was
selected for deployment in this application.
""")
