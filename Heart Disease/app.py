import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------- Load Model and Scaler --------------------
try:
    with open("heart_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    model_loaded = True
except:
    model_loaded = False

# -------------------- App UI --------------------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

st.title("🫀 AI Heart Disease Prediction System")
st.write("This AI model predicts the likelihood of **heart disease** based on medical data.")

st.markdown("---")

# -------------------- Input Fields --------------------
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])

with col2:
    restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–3) Colored by Fluoroscopy", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# -------------------- Prediction Button --------------------
st.markdown("---")
if st.button("🔍 Predict Heart Disease Risk"):
    if not model_loaded:
        st.error("⚠️ Model not found! Please train and save 'heart_model.pkl' first.")
    else:
        # Convert categorical values
        sex_val = 1 if sex == "Male" else 0

        # Create feature array
        input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])

        # Scale features
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        # Display result
        st.subheader("🩺 Prediction Result:")
        if prediction == 1:
            st.error(f"⚠️ High Risk of Heart Disease (Probability: {prob:.2f})")
        else:
            st.success(f"✅ No Heart Disease Detected (Probability: {prob:.2f})")

# -------------------- Footer --------------------
st.markdown("---")
st.caption("Developed using Machine Learning & Streamlit | © 2025")

