import streamlit as st
import joblib
import pandas as pd
import os

model_path = r"C:\Users\nagen\OneDrive\Desktop\Projects\HeartDiseasePrediction\Model\xgboost_model.pkl"

# Load the saved model
model = joblib.load(model_path)

# Function to make predictions
def predict_heart_disease(inputs):
    # Convert input values into a DataFrame
    input_df = pd.DataFrame([inputs], columns=[
        "Age", "Sex", "Chest pain type", "BP", "Cholesterol", "FBS over 120", "EKG results", 
        "Max HR", "Exercise angina", "ST depression", "Slope of ST", "Number of vessels fluro", 
        "Thallium"
    ])
    # Make prediction using the model
    prediction = model.predict(input_df)
    return prediction

# Streamlit app UI
st.title("Heart Disease Prediction")

st.write("""
This is a predictive model for heart disease detection. Please fill in the following details:
""")

# Initialize session state for input values if not already set
st.session_state.age = None
st.session_state.sex = None
st.session_state.chest_pain_type = None
st.session_state.bp = None
st.session_state.cholesterol = None
st.session_state.fbs = None
st.session_state.ekg_results = None
st.session_state.max_hr = None
st.session_state.exercise_angina = None
st.session_state.st_depression = None
st.session_state.slope_of_st = None
st.session_state.num_vessels_fluro = None
st.session_state.thallium = None

# Input fields for the user
age = st.number_input("Age", min_value=0, max_value=120)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest pain type", [1, 2, 3, 4])
bp = st.number_input("Blood Pressure", min_value=50, max_value=200)
cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600)
fbs = st.selectbox("FBS over 120 (0-NO, 1-Yes)", [0, 1])  # 0 = No, 1 = Yes
ekg_results = st.selectbox("EKG results", [0, 1, 2])
max_hr = st.number_input("Max Heart Rate", min_value=50, max_value=250)
exercise_angina = st.selectbox("Exercise angina", [0, 1])  # 0 = No, 1 = Yes
st_depression = st.number_input("ST depression", min_value=0.0, max_value=10.0)
slope_of_st = st.selectbox("Slope of ST", [1, 2, 3])
num_vessels_fluro = st.selectbox("Number of vessels fluro", [0, 1, 2, 3])
thallium = st.selectbox("Thallium", [3, 6, 7])

# Prepare inputs
inputs = {
    "Age": age,
    "Sex": 1 if sex == "Male" else 0,
    "Chest pain type": chest_pain_type,
    "BP": bp,
    "Cholesterol": cholesterol,
    "FBS over 120": fbs,
    "EKG results": ekg_results,
    "Max HR": max_hr,
    "Exercise angina": exercise_angina,
    "ST depression": st_depression,
    "Slope of ST": slope_of_st,
    "Number of vessels fluro": num_vessels_fluro,
    "Thallium": thallium
}

# Prediction button
if st.button("Predict"):
    prediction = predict_heart_disease(inputs)
    
    if prediction[0] == 0:
        st.write("No heart disease detected.")
    else:
        st.write("Heart disease detected!")
# Reset button
if st.button("Reset"):
    # Clear all input fields by setting values to default
    st.session_state.clear()