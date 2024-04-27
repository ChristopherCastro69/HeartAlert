import pickle

import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from Info import display_info_popover

st.title('Heart Disease Prediction')
# Load the whole dataset
whole_dataset = pd.read_csv('heartdisease.csv')  # Change the file path accordingly




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
    user_input['ChestPainType'] = user_input['ChestPainType'].map({'ATA': 1, 'NAP': 2, 'ASY': 0})
    user_input['FastingBS'] = user_input['FastingBS'].astype(int)
    # user_input['RestingECG'] = user_input['RestingECG'].map({'Normal': 0, 'ST': 1, 'LVH': 2})
    user_input['ExerciseAngina'] = user_input['ExerciseAngina'].map({'N': 0, 'Y': 1})
    user_input['ST_Slope'] = user_input['ST_Slope'].map({'Up': 2, 'Flat': 1, 'Down': 0})

    return user_input

display_info_popover()

# Sidebar with user input fields
st.sidebar.header('Input Features')

# Age
age = st.sidebar.slider('Age', min_value=0, max_value=100, step=1)
# Sex
sex = st.sidebar.radio('Sex', ['M', 'F'])
# Chest Pain Type
chest_pain_type = st.sidebar.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY'])

# Resting Blood Pressure
# resting_bp = st.sidebar.slider('Resting Blood Pressure (mm Hg)', min_value=90, max_value=200, step=1)

# Cholesterol
cholesterol = st.sidebar.slider('Cholesterol (mm/dl)', min_value=100, max_value=400, step=1)

# # Fasting Blood Sugar
fasting_bs = st.sidebar.radio('Fasting Blood Sugar:                                         \n(input 1 if greater than 120 mg/dl else 0)', ['0', '1'])

# fasting_bs = st.sidebar.slider('Fasting Blood Sugar', min_value=0, max_value=400)

# Resting ECG
# resting_ecg = st.sidebar.selectbox('Resting ECG Results', ['Normal', 'ST', 'LVH'])

# Maximum Heart Rate
max_hr = st.sidebar.slider('Maximum Heart Rate Achieved', min_value=50, max_value=202, step=1)
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
    if latest_user_input.empty:
        st.warning("Please fill up the latest user input data.")
    else:
        # Concatenate user input with the whole dataset
        combined_data = pd.concat([whole_dataset, latest_user_input], ignore_index=True)

        # Scale the combined_data
        mms = MinMaxScaler()
        ss = StandardScaler()

        # Fit the scaler on the whole_dataset
        mms.fit(combined_data[['Oldpeak']])
        ss.fit(combined_data[['Age', 'Cholesterol', 'MaxHR']])

        # Scale the combined_data
        combined_data['Oldpeak'] = mms.transform(combined_data[['Oldpeak']])
        combined_data[['Age', 'Cholesterol', 'MaxHR']] = ss.transform(combined_data[['Age', 'Cholesterol', 'MaxHR']])

        # Display the scaled latest_user_input
        st.subheader('Scaled Latest User Input Data')
        st.write(combined_data.iloc[-1:])  # Display only the last row, which corresponds to latest_user_input

        # Load the Model using pickle
        with open('logistic_regression_model.pkl', 'rb') as file:
            model = pickle.load(file)

        # Make predictions
        prediction = model.predict(combined_data.iloc[-1:])  # Use scaled latest_user_input

        # Display prediction
        st.subheader('Prediction')
        if prediction[0] == 1:
            st.write('Heart disease detected.')
        else:
            st.write('No heart disease detected.')

# Reset button
if st.button('Reset'):
    st.experimental_rerun()