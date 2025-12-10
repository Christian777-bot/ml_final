# -*- coding: utf-8 -*-
"""Streamlit App – Kelompok 11 (XGBoost Only)
"""

import streamlit as st
import numpy as np
import joblib

# =====================================================
# LOAD MODEL & SCALER
# =====================================================
xgb_model = joblib.load("kel11_xgb_model.sav")
scaler = joblib.load("kel11_scaler.sav")

# =====================================================
# STREAMLIT UI
# =====================================================
st.title("Obesity Level Classification – Kelompok 11")
st.markdown("""
Aplikasi ini memprediksi tingkat obesitas berdasarkan data kesehatan dan gaya hidup.
Model yang digunakan: **XGBoost (model terbaik Kelompok 11)**.
""")

st.header("Input Data")

col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.number_input("Age", 10, 65, 25)
    Height = st.number_input("Height (m)", 1.3, 2.2, 1.70)
    Weight = st.number_input("Weight (kg)", 30, 200, 70)
    FCVC = st.slider("Vegetable Consumption Frequency (FCVC)", 1.0, 3.0, 2.0)
    NCP = st.slider("Number of Main Meals (NCP)", 1.0, 4.0, 3.0)

with col2:
    CAEC = st.selectbox("Eating Between Meals (CAEC)", ["Never","Sometimes","Frequently","Always"])
    CH2O = st.slider("Water Intake (CH2O)", 1.0, 3.0, 2.0)
    FAF = st.slider("Physical Activity Frequency (FAF)", 0.0, 3.0, 1.0)
    TUE = st.slider("Time Using Technology (TUE)", 0.0, 2.0, 1.0)
    CALC = st.selectbox("Alcohol Consumption", ["Never","Sometimes","Frequently"])
    MTRANS = st.selectbox("Transportation",
        ["Public_Transportation","Automobile","Walking","Bike","Motorbike"])

family_history = st.selectbox("Family History Overweight (FHO)", [0,1])
FAVC = st.selectbox("High Caloric Food (FAVC)", [0,1])
SMOKE = st.selectbox("Smoking", [0,1])
SCC = st.selectbox("Calories Monitoring (SCC)", [0,1])

# =====================================================
# MANUAL ENCODING SESUAI PDF KELOMPOK 11
# =====================================================
gender_map = {'Male':1, 'Female':0}
caec_map   = {'Never':0, 'Sometimes':1, 'Frequently':2, 'Always':3}
calc_map   = {'Never':0, 'Sometimes':1, 'Frequently':2}
mtrans_map = {
    'Public_Transportation':0,
    'Automobile':1,
    'Walking':2,
    'Bike':3,
    'Motorbike':4
}

Gender = gender_map[Gender]
CAEC = caec_map[CAEC]
CALC = calc_map[CALC]
MTRANS = mtrans_map[MTRANS]

# =====================================================
# Binning (diperlukan karena model dilatih dengan Age_Group dan Weight_Group)
# =====================================================
def age_group_value(age):
    if   10 <= age < 20: return 0
    elif 20 <= age < 30: return 1
    elif 30 <= age < 40: return 2
    elif 40 <= age < 50: return 3
    elif 50 <= age <= 65: return 4
    return 0

def weight_group_value(weight):
    if   30 <= weight < 60: return 0
    elif 60 <= weight < 80: return 1
    elif 80 <= weight < 100: return 2
    elif 100 <= weight < 120: return 3
    elif 120 <= weight < 140: return 4
    elif weight >= 140: return 5
    return 0

Age_Group = age_group_value(Age)
Weight_Group = weight_group_value(Weight)

# =====================================================
# SUSUN FEATURE DALAM URUTAN YANG SAMA SEPERTI TRAINING
# =====================================================
input_data = np.array([[
    Gender, Age, Height, Weight,
    family_history, FAVC, FCVC, CAEC, SMOKE,
    CH2O, SCC, FAF, TUE, CALC, MTRANS,
    Age_Group, Weight_Group
]])

# Scaling untuk XGBoost
input_scaled = scaler.transform(input_data)

# Kelas label
label_map = {
    0:"Insufficient Weight",
    1:"Normal Weight",
    2:"Overweight I",
    3:"Overweight II",
    4:"Obesity I",
    5:"Obesity II",
    6:"Obesity III"
}

# =====================================================
# BUTTON PREDICT
# =====================================================
if st.button("Predict Obesity Level"):
    pred = xgb_model.predict(input_scaled)[0]
    st.subheader("Prediction Result")
    st.metric("Obesity Level", label_map[pred])
