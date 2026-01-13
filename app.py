import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ===============================
# Load Model & Preprocessor
# ===============================
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("preprocessor.pkl", "rb") as file:
    preprocessor = pickle.load(file)

# ===============================
# Streamlit Page Config
# ===============================
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="💉",
    layout="centered"
)

st.title("💊 Medical Insurance Cost Prediction App")
st.markdown("Predict **medical insurance charges** based on personal information")

st.divider()

# ===============================
# User Input Section
# ===============================
st.subheader("🧾 Enter Patient Details")

age = st.number_input("Age", min_value=18, max_value=100, value=25)

sex = st.selectbox("Sex", ["male", "female"])

bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])

smoker = st.selectbox("Smoker", ["yes", "no"])

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# ===============================
# Prediction Button
# ===============================
if st.button("🔮 Predict Insurance Cost"):

    # Create DataFrame from inputs
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region": [region]
    })

    # Transform input using preprocessor
    transformed_data = preprocessor.transform(input_data)

    # Prediction
    prediction = model.predict(transformed_data)

    st.success(f"💰 **Estimated Medical Insurance Cost:** ₹ {prediction[0]:,.2f}")

    st.balloons()

# ===============================
# Footer
# ===============================
st.divider()
st.caption("📊 Model trained using Machine Learning | Streamlit App")
