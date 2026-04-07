import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Load model
model, columns = pickle.load(open("model.pkl", "rb"))

# ---------------- SINGLE PREDICTION ----------------
def predict(gender, senior, tenure, monthly, total, contract, internet, tech, payment):

    data = {
        "SeniorCitizen": senior,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "gender": gender,
        "Contract": contract,
        "InternetService": internet,
        "TechSupport": tech,
        "PaymentMethod": payment
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    prob = model.predict_proba(df)[0][1]
    pred = model.predict(df)[0]

    if pred == 1:
        return f"⚠️ High Risk Customer\nChurn Probability: {prob*100:.2f}%"
    else:
        return f"✅ Safe Customer\nRetention Probability: {(1-prob)*100:.2f}%"

# ---------------- CSV PREDICTION ----------------
def predict_csv(file):

    df = pd.read_csv(file.name)

    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=columns, fill_value=0)

    preds = model.predict(df_encoded)
    probs = model.predict_proba(df_encoded)[:, 1]

    df["Churn"] = preds
    df["Probability"] = probs

    output_file = "results.csv"
    df.to_csv(output_file, index=False)

    return output_file

# ---------------- UI ----------------
with gr.Blocks() as app:

    gr.Markdown("# 🚀 Customer Churn Prediction System")
    gr.Markdown("Fast ML App using Gradio + Hugging Face")

    with gr.Tab("🔍 Single Prediction"):

        gender = gr.Radio(["Male","Female"], label="Gender")
        senior = gr.Radio([0,1], label="Senior Citizen")
        tenure = gr.Slider(0,72, label="Tenure")
        monthly = gr.Number(label="Monthly Charges")
        total = gr.Number(label="Total Charges")
        contract = gr.Dropdown(["Month-to-month","One year","Two year"])
        internet = gr.Dropdown(["DSL","Fiber optic","No"])
        tech = gr.Dropdown(["Yes","No"])
        payment = gr.Dropdown(["Electronic check","Mailed check","Bank transfer","Credit card"])

        output = gr.Textbox(label="Prediction")

        btn = gr.Button("Predict")

        btn.click(predict,
                  inputs=[gender, senior, tenure, monthly, total, contract, internet, tech, payment],
                  outputs=output)

    with gr.Tab("📁 Upload CSV"):

        file_input = gr.File(label="Upload CSV")
        file_output = gr.File(label="Download Results")

        btn2 = gr.Button("Run Prediction")

        btn2.click(predict_csv, inputs=file_input, outputs=file_output)

app.launch()