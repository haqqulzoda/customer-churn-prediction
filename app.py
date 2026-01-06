import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the Model and the "Skeleton" (Column Names)
model = joblib.load('models/rf.joblib')
model_columns = joblib.load('models/churn_columns.joblib')

st.title("📊 Customer Retention Engine")
st.write("Predict if a customer will leave (Churn) based on their profile.")

# --- SECTION 1: User Inputs ---
st.sidebar.header("Customer Profile")

# Numerical Inputs
tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, value=tenure * monthly_charges)

# Categorical Inputs (Dropdowns)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# --- SECTION 2: Preprocessing (The Translation Layer) ---
input_data = pd.DataFrame({
    'tenure': [tenure],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges],
    'Contract': [contract],
    'InternetService': [internet_service],
    'PaymentMethod': [payment_method],
    'PaperlessBilling': [paperless_billing],
})

# 1. Apply One-Hot Encoding (same as in notebook)
input_encoded = pd.get_dummies(input_data)


input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

# --- SECTION 3: Prediction & Thresholding ---

st.subheader("Prediction Result")

# Probability Logic
probs = model.predict_proba(input_final)[:, 1] # Probability of Churn (0.0 to 1.0)
churn_prob = float(probs[0])

# Threshold Slider (The Business Control)
threshold = st.slider("Risk Threshold (Sensitivity)", 0.0, 1.0, 0.3, 
                      help="Lower threshold = Catch more churners (High Recall). Higher = Fewer False Alarms.")

# The Final Verdict
if churn_prob >= threshold:
    st.error(f"🚨 High Risk of Churn! (Probability: {churn_prob:.2%})")
    st.write("**Recommendation:** Offer a 15% discount on the next bill or upgrade to a 1-year contract.")
else:
    st.success(f"✅ Safe Customer. (Probability: {churn_prob:.2%})")
    st.write("**Recommendation:** No immediate action needed.")

with st.expander("See Internal Data (Debug)"):
    st.write(input_final)