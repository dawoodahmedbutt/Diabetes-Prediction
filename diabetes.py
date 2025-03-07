import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

# Set page configuration - this changes the browser tab title
st.set_page_config(
    page_title="Diabetes Model",
    page_icon="🩺",
    layout="wide"
)

# Set environment variables to disable GPU
## '3' means "only show errors" (suppress info, warnings, and debug messages)
## This disables GPU usage for TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging



# load the model
@st.cache_resource
def load_model():
    return pickle.load(open('model.pkl', 'rb'))

# Load the scaler
@st.cache_resource
def load_scaler():
    return pickle.load(open('scaler.pkl', 'rb'))

# Load models and scaler
model = load_model()
scaler = load_scaler()

# Function to make predictions
def make_prediction(input_data):
# Convert the input data to a pandas DataFrame
    input_df = pd.DataFrame([input_data], columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'
])

    # Output the user's input
    st.write("### User Input:")

    # Display the input as a DataFrame
    st.dataframe(input_df)

    # Apply the scaling transformation
    scaled_data = scaler.transform(input_df)

    # Make the prediction
    prediction = model.predict(scaled_data)
    return prediction

# Streamlit UI
st.title('Diabetes Prediction')

# Display an image
# st.image('Diabetes_pic.jpg', use_container_width=True)

st.write("""
Please fill in the information below, and we'll predict if you might be at risk of diabetes based on the data you provide.  
Note: All fields are important to give the most accurate prediction.
""")

# Collect input features from the user

pregnancies = st.number_input(
    'Number of Pregnancies',
    min_value=0,
    max_value=20,
    value=0,
    step=1,
    help="Enter the number of times you have been pregnant."
)

age = st.number_input(
    'Age',
    min_value=1,
    max_value=100,
    value=30,
    step=1,
    help="Enter your age in years."
)

glucose = st.number_input(
    'Glucose Level (mg/dL)',
    min_value=0,
    max_value=300,
    value=120,
    step=1,
    help="Enter your blood glucose level from your most recent test."
)

blood_pressure = st.number_input(
    'Blood Pressure (mm Hg)',
    min_value=0,
    max_value=200,
    value=70,
    step=1,
    help="Enter your diastolic blood pressure."
)

skin_thickness = st.number_input(
    'Skin Thickness (mm)',
    min_value=0,
    max_value=100,
    value=20,
    step=1,
    help="Enter your triceps skin fold thickness."
)

insulin = st.number_input(
    'Insulin Level (mu U/ml)',
    min_value=0,
    max_value=1000,
    value=79,
    step=1,
    help="Enter your 2-Hour serum insulin level."
)

bmi = st.number_input(
    'BMI (kg/m²)',
    min_value=0.0,
    max_value=70.0,
    value=25.0,
    step=0.1,
    help="Enter your Body Mass Index (BMI)."
)

diabetes_pedigree = st.number_input(
    'Diabetes Pedigree Function',
    min_value=0.0,
    max_value=3.0,
    value=0.5,
    step=0.01,
    help="Enter your diabetes pedigree function value (a function that scores likelihood of diabetes based on family history)."
)


# Convert inputs into appropriate format for prediction
input_data = [
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
]

# Button to trigger prediction
if st.button('Predict'):
# Make prediction\
    prediction = make_prediction(input_data)

# Setting threshold for probability output
    if prediction > 0.5:
        st.write("The model predicts: **Diabetes Risk**")
        st.write("It's recommended to consult with a healthcare professional for further assessment.")
    else:
        st.write("The model predicts: **No Diabetes Risk**")
        st.write("You seem to be at a lower risk based on the provided information.")
