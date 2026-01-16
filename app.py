import streamlit as st
import pandas as pd
import pickle
import numpy as np


# PAGE CONFIG
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    
    layout="centered"
)


# LOAD MODEL

with open("model.pkl", "rb") as f:
    model = pickle.load(f)



st.title("Medical Insurance Cost Prediction App")
st.write("Enter details to predict insurance cost")

st.divider()

age = st.number_input("Age", 18, 100, 25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
children = st.selectbox("Children", [0, 1, 2, 3, 4, 5])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)


# MANUAL ENCODING 

sex_encoded = 1 if sex == "male" else 0
smoker_encoded = 1 if smoker == "yes" else 0

region_mapping = {
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3
}

region_encoded = region_mapping[region]


# PREDICTION

if st.button("Predict Insurance Cost"):

    input_data = pd.DataFrame([[
        age,
        sex_encoded,
        bmi,
        children,
        smoker_encoded,
        region_encoded
    ]], columns=[
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region"
    ])

    prediction = model.predict(input_data)

    st.success(
        f" Predicted Insurance Cost: ₹ {np.round(prediction[0], 2):,}"
    )
