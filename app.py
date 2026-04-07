import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model + columns
model, columns = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Churn AI", layout="wide")

# ---------------- THEME ----------------
theme = st.sidebar.radio("Theme", ["Dark", "Light"])

if theme == "Dark":
    st.markdown("<style>.stApp {background-color:#0e1117; color:white;}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>.stApp {background-color:white; color:black;}</style>", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🚀 Customer Churn Intelligence System")
st.write("Predict churn using ML with batch & file support")

tab1, tab2, tab3 = st.tabs(["Single", "Batch", "Upload CSV"])

# =================================================
# 🔹 SINGLE PREDICTION
# =================================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        tenure = st.slider("Tenure", 0, 72)
        monthly = st.number_input("Monthly Charges", 0.0, 200.0)
        total = st.number_input("Total Charges", 0.0, 10000.0)
        contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        internet = st.selectbox("Internet", ["DSL","Fiber optic","No"])
        tech = st.selectbox("Tech Support", ["Yes","No"])
        payment = st.selectbox("Payment", ["Electronic check","Mailed check","Bank transfer","Credit card"])

        if st.button("Predict"):

            input_dict = {
                "SeniorCitizen": senior,
                "tenure": tenure,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }

            df = pd.DataFrame([input_dict])

            # Add categorical columns
            df["gender"] = gender
            df["Contract"] = contract
            df["InternetService"] = internet
            df["TechSupport"] = tech
            df["PaymentMethod"] = payment

            # One-hot encoding
            df = pd.get_dummies(df)

            # Match training columns
            df = df.reindex(columns=columns, fill_value=0)

            pred = model.predict(df)[0]
            prob = model.predict_proba(df)[0][1]

            with col2:
                st.metric("Churn Probability", f"{prob*100:.2f}%")

                if pred == 1:
                    st.error("High Risk 🚨")
                else:
                    st.success("Safe Customer ✅")

# =================================================
# 🔹 BATCH PREDICTION
# =================================================
with tab2:
    st.write("Enter multiple customers")

    df = st.data_editor(pd.DataFrame(columns=columns))

    if st.button("Run Batch"):
        preds = model.predict(df)
        probs = model.predict_proba(df)[:,1]

        df["Churn"] = preds
        df["Probability"] = probs

        st.dataframe(df)

# =================================================
# 🔹 CSV UPLOAD
# =================================================
with tab3:
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

        preds = model.predict(df_encoded)
        probs = model.predict_proba(df_encoded)[:,1]

        df["Churn"] = preds
        df["Probability"] = probs

        st.dataframe(df)

        st.download_button("Download Results", df.to_csv(index=False), "results.csv")