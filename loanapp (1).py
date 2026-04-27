
# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd
import sklearn  # This is needed for the pickle file to load!

# Load the trained model
# --- Put the Model in Drive First---
with open("my_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title for the app
st.markdown(
    "<h1 style='text-align: center; background-color: #ffcccc; padding: 10px; color: #cc0000;'><b>Loan Data Analyzer</b></h1>",
    unsafe_allow_html=True
)

st.markdown("""
<style>
    /* Custom CSS for blue sliders */
    .stSlider .st-bj { /* Filled track */
        background-color: #1E90FF !important; /* Dodger Blue */
    }
    .stSlider .st-bk { /* Slider thumb */
        background-color: #1E90FF !important; /* Dodger Blue */
        border-color: #1E90FF !important;
    }
    .stSlider .st-bi { /* Unfilled track */
        background-color: #ADD8E6 !important; /* Light blue */
    }
</style>
""", unsafe_allow_html=True)

# Numeric inputs
st.header("Enter Loan Applicant's Details")

# Input fields for numeric values based on the final model's features
requested_loan_amount = st.slider("Requested Loan Amount", min_value=5000, max_value=125000, step=1000, value=50000)
fico_score = st.slider("FICO Score", min_value=385, max_value=850, step=1, value=650)
monthly_gross_income = st.slider("Monthly Gross Income", min_value=-2559, max_value=14005, step=100, value=5000)
monthly_housing_payment = st.slider("Monthly Housing Payment", min_value=300, max_value=3300, step=100, value=1500)
ever_bankrupt_or_foreclose = st.selectbox("Ever Bankrupt or Foreclose", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")

# Categorical inputs with options based on the final model's features
reason = st.selectbox("Reason", ['cover_an_unexpected_cost', 'credit_card_refinancing', 'home_improvement', 'major_purchase', 'other', 'debt_conslidation'])
employment_status = st.selectbox("Employment Status", ['full_time', 'part_time', 'unemployed'])
employment_sector = st.selectbox("Employment Sector", ['consumer_discretionary', 'information_technology', 'energy', 'consumer_staples', 'communication_services', 'materials', 'utilities', 'real_estate', 'health_care', 'industrials', 'financials', 'Unknown'])
lender = st.selectbox("Lender", ['B', 'A', 'C'])

# Create the input data as a DataFrame
input_data = pd.DataFrame({
    "Requested_Loan_Amount": [requested_loan_amount],
    "FICO_score": [fico_score],
    "Monthly_Gross_Income": [monthly_gross_income],
    "Monthly_Housing_Payment": [monthly_housing_payment],
    "Ever_Bankrupt_or_Foreclose": [ever_bankrupt_or_foreclose],
    "Reason": [reason],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Lender": [lender]
})

# --- Prepare Data for Prediction ---
# 1. One-hot encode the user's input.
input_data_encoded = pd.get_dummies(input_data, columns=['Reason', 'Employment_Status', 'Employment_Sector', 'Lender'])

# 2. Add any "missing" columns the model expects (fill with 0).
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# 3. Reorder/filter columns to exactly match the model's training data.
input_data_encoded = input_data_encoded[model_columns]

# Predict button
if st.button("Evaluate Loan"):
    # Predict using the loaded model
    prediction = model.predict(input_data_encoded)[0]

    # Display result (Corrected interpretation: 1=Approved, 0=Denied)
    if prediction == 1:
        st.markdown("<h3>The prediction is: <b>Approved Loan</b> ✅</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3>The prediction is: <b>Denied Loan</b> ❌</h3>", unsafe_allow_html=True)
