import streamlit as st
import pandas as pd
import pickle
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler

st.title('Heart Disease Prediction')

# Function to initialize or retrieve the user input DataFrame from session state
def initialize_user_input():
    if 'user_input' not in st.session_state:
        st.session_state.user_input = pd.DataFrame({
            'Age': [],
            'Sex': [],
            'ChestPainType': [],
            # 'RestingBP': [],
            'Cholesterol': [],
            'FastingBS': [],
            # 'RestingECG': [],
            'MaxHR': [],
            'ExerciseAngina': [],
            'Oldpeak': [],
            'ST_Slope': []
        })
    return st.session_state.user_input

# Function to save user input and refresh DataFrame
def save_and_refresh(user_input):
    st.session_state.user_input = user_input
    st.experimental_rerun()

# Function to preprocess user input
def preprocess_input(user_input):
    # Convert categorical variables to numeric representations
    user_input['Sex'] = user_input['Sex'].map({'M': 1, 'F': 0})
    user_input['ChestPainType'] = user_input['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2})
    user_input['FastingBS'] = user_input['FastingBS'].astype(int)
    # user_input['RestingECG'] = user_input['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
    user_input['ExerciseAngina'] = user_input['ExerciseAngina'].map({'N': 0, 'Y': 1})
    user_input['ST_Slope'] = user_input['ST_Slope'].map({'Up': 0, 'Flat': 1, 'Down': 2})

    return user_input

# Sidebar with user input fields
st.sidebar.header('Input Features')

# Age
age = st.sidebar.slider('Age', min_value=-2.0, max_value=2.0, step=0.01)

# Sex
sex = st.sidebar.radio('Sex', ['M', 'F'])

# Chest Pain Type
chest_pain_type = st.sidebar.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY'])

# Resting Blood Pressure
# resting_bp = st.sidebar.slider('Resting Blood Pressure (mm Hg)', min_value=90, max_value=200, step=1)

# Cholesterol
cholesterol = st.sidebar.slider('Cholesterol (mm/dl)', min_value=-2.0, max_value=4.0, step=0.01)

# Fasting Blood Sugar
fasting_bs = st.sidebar.radio('Fasting Blood Sugar > 120 mg/dl', ['0', '1'])

# Resting ECG
# resting_ecg = st.sidebar.selectbox('Resting ECG Results', ['Normal', 'ST', 'LVH'])

# Maximum Heart Rate
max_hr = st.sidebar.slider('Maximum Heart Rate Achieved', min_value=-3.0, max_value=3.0, step=0.01)

# Exercise-Induced Angina
exercise_angina = st.sidebar.radio('Exercise-Induced Angina', ['N', 'Y'])

# Oldpeak (ST Depression)
oldpeak = st.sidebar.slider('Oldpeak (ST Depression)', min_value=0.0, max_value=6.2, step=0.01)

# ST Slope
st_slope = st.sidebar.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

# Save button
if st.sidebar.button('Save'):
    # Create DataFrame with user inputs
    user_input = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain_type],
        # 'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        # 'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    # Preprocess user input
    user_input = preprocess_input(user_input)


    # Display the latest DataFrame from user input
    st.subheader('Latest User Input Data')
    st.write(user_input)

    # Save user input to session state
    save_and_refresh(user_input)

# Display the latest DataFrame from user input
latest_user_input = initialize_user_input()
st.subheader('Latest User Input Data')
st.write(latest_user_input)


# Predict button
if st.button('Predict'):


    # Load the Model using pickle
    with open('logistic_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Make predictions
    prediction = model.predict(latest_user_input)

    # Display prediction
    st.subheader('Prediction')
    if prediction[0] == 1:
        st.write('Heart disease detected.')
    else:
        st.write('No heart disease detected.')

    # # Print the scaled input values
    # print("Scaled Input Values:")
    # print(latest_user_input)
    #
    # # Display the scaled input table
    # st.subheader('Scaled Input Data')
    # st.write(latest_user_input)

