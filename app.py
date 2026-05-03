import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open("churn_model.pkl", "rb"))

st.title("Customer Churn Predictor")

tenure = st.number_input("Tenure")
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")

if st.button("Predict"):
    data = pd.DataFrame([[tenure, monthly, total]],
                        columns=["tenure","MonthlyCharges","TotalCharges"])

    result = model.predict(data)[0]

    if result == 1:
        st.error("Customer WILL CHURN")
    else:
        st.success("Customer will NOT churn")
