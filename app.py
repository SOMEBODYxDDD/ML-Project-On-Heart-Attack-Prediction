import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="Heart Disease Risk Assessment System",
    layout="wide"
)
model = joblib.load('Heart_disease_model.pkl')
age_map = {
    '18-24': 0, '25-29': 1, '30-34': 2, '35-39': 3, 
    '40-44': 4, '45-49': 5, '50-54': 6, '55-59': 7, 
    '60-64': 8, '65-69': 9, '70-74': 10, '75-79': 11, 
    '80+': 12
}
health_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}
checkup_map = {
    'Within the past year': 0, 'Within the past 2 years': 1, 
    'Within the past 5 years': 2, '5 or more years ago': 3
}
teeth_map = {'None': 0, '1-5': 1, '6+': 2, 'All': 3}
smoke_map = {"Never smoked": 0, "Quit smoking": 1, "Occasional smoker": 2, "Daily smoker": 3}
st.sidebar.header(" Patient Information")
def get_user_input():
    st.sidebar.subheader("1. Basic Vitals")
    age_label = st.sidebar.selectbox("Age Category", list(age_map.keys()), index=7)
    height = st.sidebar.number_input("Height (Meters)", 1.40, 2.20, 1.70)
    weight = st.sidebar.number_input("Weight (Kg)", 30.0, 200.0, 80.0)
    bmi = weight / (height ** 2)
    st.sidebar.info(f"Calculated BMI: {bmi:.2f}")
    st.sidebar.subheader("2. Health Status & Sleep")
    gen_health_label = st.sidebar.select_slider("General Health", options=list(health_map.keys()), value='Good')
    phys_days = st.sidebar.slider("Physical Health Days (Past 30)", 0, 30, 0)
    ment_days = st.sidebar.slider("Mental Health Days (Past 30)", 0, 30, 0)
    sleep_hours = st.sidebar.slider("Average Sleep Hours", 3, 12, 7)
    st.sidebar.subheader("3. Lifestyle")
    smoke_label = st.sidebar.selectbox("Smoking Status", list(smoke_map.keys()))
    alcohol = st.sidebar.radio("Alcohol Consumption", ["No", "Yes"], horizontal=True)
    checkup_label = st.sidebar.selectbox("Last Checkup", list(checkup_map.keys()))
    st.sidebar.subheader("4. Medical History")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        had_stroke = st.checkbox("Stroke")
        had_diabetes = st.checkbox("Diabetes")
        had_angina = st.checkbox("Angina")
        had_copd = st.checkbox("COPD")
    with col2:
        had_asthma = st.checkbox("Asthma")
        had_kidney = st.checkbox("Kidney Disease")
        had_skin = st.checkbox("Skin Cancer")
        had_arthritis = st.checkbox("Arthritis")
    chest_scan = st.radio("Chest CT/Scan?", ["No", "Yes"], horizontal=True)
    teeth_label = st.selectbox("Teeth Removed", ["None", "1-5", "6+", "All"])
    st.sidebar.subheader("5. Physical Difficulties")
    diff_walk = st.checkbox("Walking")
    diff_dress = st.checkbox("Dressing/Bathing")
    diff_errand = st.checkbox("Running Errands")
    diff_conc = st.checkbox("Concentrating")
    frailty_score = sum([diff_walk, diff_dress, diff_errand, diff_conc])
    comorbidity_score = sum([had_stroke, had_diabetes, had_angina, had_copd, 
                             had_asthma, had_kidney, had_skin, had_arthritis])
    data = {
        'Frailty_Score': frailty_score,
        'ChestScan': 1 if chest_scan == "Yes" else 0,
        'HadStroke': 1 if had_stroke else 0,
        'PhysicalHealthDays': phys_days,
        'MentalHealthDays': ment_days,
        'Comorbidity_Score': comorbidity_score,
        'RemovedTeeth': teeth_map[teeth_label],
        'SmokerStatus': smoke_map[smoke_label],
        'SleepHours': sleep_hours,
        'GeneralHealth': health_map[gen_health_label],
        'WeightInKilograms': weight,
        'State_Risk_Score': 0,
        'AgeCategory': age_map[age_label],
        'HeightInMeters': height,
        'BMI': bmi
    }
    return pd.DataFrame(data, index=[0])
input_df = get_user_input()
st.title("Heart Disease Risk Assessment System")
st.markdown("Based on LightGBM Machine Learning Model & CDC Health Data")
st.divider()
with st.expander("View Input Data Debug"):
    st.dataframe(input_df)
if st.button("Assess Risk", type="primary", use_container_width=True):
    if model:
        expected_cols = [
            'Frailty_Score', 'ChestScan', 'HadStroke', 'PhysicalHealthDays', 
            'MentalHealthDays', 'Comorbidity_Score', 'RemovedTeeth', 
            'SmokerStatus', 'SleepHours', 'GeneralHealth', 'WeightInKilograms', 
            'State_Risk_Score', 'AgeCategory', 'HeightInMeters', 'BMI'
        ]
        
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_cols]
        prediction_proba = model.predict_proba(input_df)[:, 1][0]
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Risk Probability", f"{prediction_proba:.1%}")
        with col2:
            st.write("#### Risk Level")
            st.progress(prediction_proba)
            if prediction_proba > 0.5:
                st.markdown(f"<div class='big-font risk-high'> High Risk</div>", unsafe_allow_html=True)
                st.warning("Recommendation: Please consult a cardiologist immediately.")
            elif prediction_proba > 0.2:
                st.markdown(f"<div class='big-font' style='color:orange'> Moderate Risk</div>", unsafe_allow_html=True)
                st.info("Recommendation: Monitor your health and improve lifestyle habits.")
            else:
                st.markdown(f"<div class='big-font risk-low'> Low Risk</div>", unsafe_allow_html=True)
                st.success("Recommendation: Maintain a healthy lifestyle.")
        st.divider()
        st.subheader("🔍 Model Interpretation")
        st.caption("Key contributing factors based on your input:")
        top_factors = []
        if input_df['BMI'][0] > 25:
            top_factors.append(("High BMI", "Overweight increases cardiovascular load."))
        if input_df['AgeCategory'][0] >= 8:
            top_factors.append(("Age", "Advanced age is a primary risk factor."))
        if input_df['GeneralHealth'][0] <= 1:
            top_factors.append(("Poor General Health", "Self-reported health correlates strongly with risk."))
        if input_df['SmokerStatus'][0] >= 2:
            top_factors.append(("Smoking", "Smoking is a major cause of heart disease."))
        if input_df['PhysicalHealthDays'][0] > 10:
            top_factors.append(("Physical Discomfort", "Frequent physical illness days reported."))
        if input_df['HadStroke'][0] == 1:
            top_factors.append(("Stroke History", "Prior stroke significantly increases baseline risk."))
        if top_factors:
            for factor, reason in top_factors:
                st.markdown(f"- **{factor}**: {reason}")
        else:
            st.write("No single high-risk factor detected. Risk may be due to a combination of minor factors.")
        with st.expander("Compare Key Metrics"):
            chart_data = pd.DataFrame({
                "Your Value": [input_df['BMI'][0], input_df['PhysicalHealthDays'][0], input_df['MentalHealthDays'][0]],
                "Healthy Benchmark": [24.0, 2.0, 2.0]
            }, index=["BMI", "Physical Illness Days", "Mental Illness Days"])
            st.bar_chart(chart_data)