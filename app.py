# app.py
import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")  # If you saved your scaler in preprocessing

st.title("💳 Transaction Fraud Detection App")
st.write("Enter transaction details to predict if a transaction is fraudulent.")

# Inputs — adjust according to dataset columns
time = st.number_input("Transaction Time", min_value=0.0)
amount = st.number_input("Transaction Amount", min_value=0.0)

# Example for PCA features V1-V28
# For simplicity, we can set them to zero for testing, or allow input fields for all
features = [time] + [0]*28 + [amount]  # [Time, V1-V28 zeros, Amount]

if st.button("Predict"):
    # Scale features
    input_scaled = scaler.transform([features])
    # Predict
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")
