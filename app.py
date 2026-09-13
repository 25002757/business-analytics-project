
import streamlit as st
import pandas as pd

st.title("Customer Churn Prediction")

st.header("Business Problem")

st.write("Customer churn is an important business problem because losing customers can affect revenue and customer retention.")

st.header("Objective")

st.write("The objective of this project is to develop a machine learning model that predicts whether a customer is likely to churn.")

st.header("Dataset Information")

st.write("The dataset contains 10,205 customer records and 16 variables related to customer demographics, usage behaviour, satisfaction, complaints, orders, and churn.")

df = pd.read_csv("ecommerce_churn_data.csv")

st.header("Data Insights")

st.subheader("Churn Distribution")

churn_counts = df["Churn"].value_counts()

st.bar_chart(churn_counts)

st.subheader("Customer Churn by Satisfaction Score")

satisfaction_churn = df.groupby(["SatisfactionScore", "Churn"]).size().unstack(fill_value=0)

st.bar_chart(satisfaction_churn)

st.subheader("Customer Churn by Complaint Status")

complaint_churn = df.groupby(["Complain", "Churn"]).size().unstack(fill_value=0)

st.bar_chart(complaint_churn)

st.subheader("Customer Tenure by Churn Status")

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(7,4))

sns.boxplot(data=df, x="Churn", y="Tenure", ax=ax)

ax.set_xlabel("Churn (0 = No, 1 = Yes)")
ax.set_ylabel("Tenure")

st.pyplot(fig)

st.subheader("Days Since Last Order by Churn Status")

fig, ax = plt.subplots(figsize=(7,4))

sns.boxplot(data=df, x="Churn", y="DaySinceLastOrder", ax=ax)

ax.set_xlabel("Churn (0 = No, 1 = Yes)")
ax.set_ylabel("Days Since Last Order")

st.pyplot(fig)

import joblib

loaded_model = joblib.load("model.pkl")

st.header("Prediction")

age = st.number_input("Age", min_value=18, max_value=100, value=30)

tenure = st.number_input("Tenure", min_value=0, max_value=100, value=10)

satisfaction = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)

complain = st.selectbox("Complain", [0, 1])

order_count = st.number_input("Order Count", min_value=0, max_value=300, value=10)

gender = st.selectbox("Gender", ["Female", "Male", "Other"])

login_device = st.selectbox(
    "Preferred Login Device",
    ["Desktop", "Mobile"]
)

city_tier = st.number_input("City Tier", min_value=1, max_value=3, value=1)

warehouse_to_home = st.number_input(
    "Warehouse To Home",
    min_value=0.0,
    value=10.0
)

hours_spent = st.number_input(
    "Hours Spent On App",
    min_value=0.0,
    value=3.0
)

devices = st.number_input(
    "Number Of Devices Registered",
    min_value=0,
    value=2
)

addresses = st.number_input(
    "Number Of Address",
    min_value=0,
    value=2
)

days_last_order = st.number_input(
    "Days Since Last Order",
    min_value=0,
    value=5
)

cashback = st.number_input(
    "Cashback Amount",
    min_value=0.0,
    value=150.0
)

if st.button("Predict Churn"):

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "PreferredLoginDevice": login_device,
        "CityTier": city_tier,
        "WarehouseToHome": warehouse_to_home,
        "HoursSpentOnApp": hours_spent,
        "NumberOfDevicesRegistered": devices,
        "SatisfactionScore": satisfaction,
        "NumberOfAddress": addresses,
        "Complain": complain,
        "OrderCount": order_count,
        "DaySinceLastOrder": days_last_order,
        "CashbackAmount": cashback
    }])

    prediction = loaded_model.predict(input_data)[0]
    probability = loaded_model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("Prediction: Customer is likely to churn")
    else:
        st.success("Prediction: Customer is unlikely to churn")

    st.write(f"Churn Probability: {probability:.2%}")
