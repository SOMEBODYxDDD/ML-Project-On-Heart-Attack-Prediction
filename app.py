import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
st.set_page_config(
    page_title="Heart Disease Risk Prediction System",
    layout="wide"
)
model= CatBoostClassifier()
model.load_model('heart_disease_model.cbm')
age_map = {
    '18-24': 21, '25-29': 27, '30-34': 32, '35-39': 37, 
    '40-44': 42, '45-49': 47, '50-54': 52, '55-59': 57, 
    '60-64': 62, '65-69': 67, '70-74': 72, '75-79': 77, 
    '80+': 80
}
health_map = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Very good': 4, 'Excellent': 5}
checkup_map = {
    'Within the past year': 1, 'Within the past 2 years': 2, 'Within the past 5 years': 3, '5 or more years ago': 4
}
st.sidebar.header(" Patient Information Input")

def user_input_features():
    st.sidebar.subheader("1. Basic Vitals")
    age_cat = st.sidebar.selectbox("Age Group", list(age_map.keys()), index=7)
    height = st.sidebar.number_input("Height (Meters)", 1.40, 2.20, 1.70)
    weight = st.sidebar.number_input("Weight (Kilograms)", 30.0, 200.0, 80.0)
    gen_health = st.sidebar.select_slider("Self-Rated General Health", options=list(health_map.keys()), value='Good')
    st.sidebar.subheader("2. Lifestyle & Checkups")
    smoker = st.sidebar.radio("Smoking Status", ["Never smoked", "Quit smoking", "Occasional smoker", "Daily smoker"])
    alcohol = st.sidebar.radio("Alcohol Consumption", ["No", "Yes"])
    checkup = st.sidebar.selectbox("Last Checkup Time", list(checkup_map.keys()))
    teeth = st.sidebar.selectbox("Tooth Loss Status", ["None", "1-5 teeth removed", "6+ teeth removed", "All teeth removed"])
    st.sidebar.subheader("3. Medical History")
    st.sidebar.caption("Please check confirmed conditions:")
    chronic_conditions = {
        'HadAsthma': st.sidebar.checkbox("Asthma"),
        'HadSkinCancer': st.sidebar.checkbox("Skin Cancer"),
        'HadCOPD': st.sidebar.checkbox("Chronic Obstructive Pulmonary Disease (COPD)"),
        'HadDepressiveDisorder': st.sidebar.checkbox("Depressive Disorder"),
        'HadKidneyDisease': st.sidebar.checkbox("Kidney Disease"),
        'HadArthritis': st.sidebar.checkbox("Arthritis"),
        'HadDiabetes': st.sidebar.checkbox("Diabetes"),
        'HadAngina': st.sidebar.checkbox("Angina"),
        'HadStroke': st.sidebar.checkbox("Stroke")
    }
    st.sidebar.subheader("4. Physical Function")
    st.sidebar.caption("Do you have difficulty with the following:")
    diff_conditions = {
        'DifficultyWalking': st.sidebar.checkbox("Difficulty Walking"),
        'DifficultyDressingBathing': st.sidebar.checkbox("Difficulty Dressing/Bathing"),
        'DifficultyErrands': st.sidebar.checkbox("Difficulty Running Errands Independently"),
        'BlindOrVisionDifficulty': st.sidebar.checkbox("Blindness or Vision Difficulty"),
        'DeafOrHardOfHearing': st.sidebar.checkbox("Deafness or Hard of Hearing"),
        'DifficultyConcentrating': st.sidebar.checkbox("Difficulty Concentrating")
    }
    chest_scan = st.sidebar.radio("Have you had a Chest CT/Scan?", ["No", "Yes"])
    comorbidity_score = sum(chronic_conditions.values())
    frailty_score = sum(diff_conditions.values())
    smoke_val = 0
    if smoker == "Quit smoking": smoke_val = 1
    elif smoker == "Occasional smoker": smoke_val = 2
    elif smoker == "Daily smoker": smoke_val = 3
    teeth_val = 0
    if teeth == "1-5 teeth removed": teeth_val = 1
    elif teeth == "6+ teeth removed": teeth_val = 2
    elif teeth == "All teeth removed": teeth_val = 3
    data = {
        'WeightInKilograms': weight,
        'LastCheckupTime': checkup_map[checkup],
        'AlcoholDrinkers': 1 if alcohol == "Yes" else 0,
        'SmokerStatus': smoke_val,
        'Frailty_Score': frailty_score,
        'HadDiabetes': 1 if chronic_conditions['HadDiabetes'] else 0,
        'Comorbidity_Score': comorbidity_score,
        'HeightInMeters': height,
        'RemovedTeeth': teeth_val,
        'Cluster_0': 0,
        'HadStroke': 1 if chronic_conditions['HadStroke'] else 0,
        'ChestScan': 1 if chest_scan == "Yes" else 0,
        'GeneralHealth': health_map[gen_health],
        'AgeCategory': age_map[age_cat],
        'HadAngina': 1 if chronic_conditions['HadAngina'] else 0
    }
    
    return pd.DataFrame(data, index=[0])
st.title(" Heart Disease Risk Intelligent Assessment System")
st.markdown("""
This application is based on a **CatBoost Machine Learning Model**, trained using CDC health data.
It predicts the risk of heart disease based on the patient's vitals, lifestyle habits, and medical history.
""")
input_df = user_input_features()
with st.expander("View Current Patient Data Summary"):
    st.dataframe(input_df)
if st.button("Start Assessment", type="primary"):
    if model:
        prediction_proba = model.predict_proba(input_df)[:, 1][0]
        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Risk Probability", value=f"{prediction_proba:.1%}")
        with col2:
            # Dynamic progress bar
            st.write("Risk Level Visualization")
            st.progress(prediction_proba)
            threshold = 0.175 
            if prediction_proba > 0.5:
                st.error(f"**High Risk Warning** (Probability > 50%)")
                st.write("Recommendation: Please consult a cardiologist immediately for detailed examination.")
            elif prediction_proba > threshold:
                st.warning(f" **Moderate Risk** (Probability > {threshold*100}%)")
                st.write("Recommendation: Your risk is above the screening threshold. We recommend monitoring cardiovascular health and having regular checkups.")
            else:
                st.success("**Low Risk**")
                st.write("Recommendation: Maintain a healthy lifestyle!")
        st.subheader("Key Risk Factors Analysis")
        st.info("The following factors contributed most to this prediction (based on model Top Features):")
        risk_factors = []
        if input_df['AgeCategory'][0] > 60: risk_factors.append("Older age")
        if input_df['Comorbidity_Score'][0] >= 2: risk_factors.append(f"High Comorbidity Score ({input_df['Comorbidity_Score'][0]} concurrent conditions)")
        if input_df['SmokerStatus'][0] > 1: risk_factors.append("Smoking habit")
        if input_df['GeneralHealth'][0] <= 2: risk_factors.append("Poor self-rated general health")
        if risk_factors:
            for factor in risk_factors:
                st.write(f"- {factor}")
        else:
            st.write("No significant high-risk factors detected.")