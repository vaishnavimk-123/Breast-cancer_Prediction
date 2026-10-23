import streamlit as st
import pandas as pd
import joblib

# Load the trained model and selected feature names
model = joblib.load("breast_cancer_model.pkl")
selected_features = joblib.load("selected_features.pkl")

st.title("Breast Cancer Prediction")
st.write("Enter the selected tumor feature measurements.")

with st.form("prediction_form"):
    values = {}

    for feature in selected_features:
        values[feature] = st.number_input(
            feature,
            value=0.0,
            format="%.6f"
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Make input in the format expected by the model
        new_input = pd.DataFrame([values])

        prediction = model.predict(new_input)[0]

        # Diagnosis mapping from your notebook
        if prediction == 0:
            st.success("Model prediction: Benign")
        elif prediction == 1:
            st.warning("Model prediction: Malignant")
        else:
            st.write("Model output:", prediction)

        st.caption(
            "Educational model prediction only, not a medical diagnosis."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")
