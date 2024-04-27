import streamlit as st

# Define the function to display the popover with the markdown text
def display_info_popover():
    with st.popover("Info"):
        info_text = """
        ### Problem Statement :

        Cardiovascular diseases (CVDs) are the top cause of global death, claiming about 17.9 million lives yearly, accounting for 31% of all deaths.

        Machine learning models can assist in early detection and management of cardiovascular disease for those at risk, including individuals with hypertension, diabetes, hyperlipidemia, or existing heart conditions.

        The dataset includes 9 features to predict heart disease risk.

        ### Aim
        - To classify / predict whether a patient is prone to heart failure depending on multiple attributes.
        - It is a **binary classification** with multiple numerical and categorical features.

        ### Dataset Attributes

        - **Age** : age of the patient [years]
        - **Sex** : sex of the patient [M: Male, F: Female]
        - **ChestPainType** : chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
        - **Cholesterol** : serum cholesterol [mm/dl]
        - **FastingBS** : fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
        - **MaxHR** : maximum heart rate achieved [Numeric value between 60 and 202]
        - **ExerciseAngina** : exercise-induced angina [Y: Yes, N: No]
        - **Oldpeak** : oldpeak = ST [Numeric value measured in depression]
        - **ST_Slope** : the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
        - **HeartDisease** : output class [1: heart disease, 0: Normal]
        
        
        ### Disclaimer: 
        
        This model is a preliminary version developed for heart disease prediction and has an accuracy rate of 89.51%. It is important to note that while the model shows promising results, it is still undergoing development and training to improve its accuracy and reliability.
        
        ### Use with Caution: 
        
        Users are advised to interpret the model predictions with caution and not solely rely on them for medical diagnosis or treatment decisions. The predictions provided by the model are based on statistical analysis and machine learning algorithms and should be considered as supportive information rather than conclusive evidence of a medical condition.
        """
        st.markdown(info_text)


