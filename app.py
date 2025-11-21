import numpy as np
import pandas as pd
import streamlit as st
import joblib as jb

rf = jb.load("Random_Forest.pkl")
st.title("Self Harm Predictor Dashboard")

Gender = st.selectbox("Gender",["Male","Female"])
Heart_Rate = st.number_input("Heart Rate",min_value=40,max_value=120,value="min")
Systolic = st.number_input("Systolic Blood Pressure",min_value=70,max_value=200,value="min")
Diastolic = st.number_input("Diastolic Blood Pressure",min_value=50,max_value=120,value="min")
Stress = st.number_input("Biosensor Stress Level",min_value=1,max_value=10,value="min")
Self = st.number_input("Self Stress Report",min_value=1,max_value=10,value="min")
Physical = st.selectbox("Physical Activity",["High","Moderate","Low"])
Sleep = st.selectbox("Sleep Quality",["Good","Moderate","Poor"])
Mood = st.selectbox("Mood",["Happy","Stressed","Neutral"])
Work = st.number_input("Working Hours in a week",min_value=1,max_value=100,value="min")

Input_Data = pd.DataFrame({
                           "Gender":[1 if Gender=="Male" else 0],
                           "Heart_Rate":Heart_Rate,
                           "Blood_Pressure_Systolic":Systolic,
                           "Blood_Pressure_Diastolic":Diastolic,
                           "Stress_Level_Biosensor":Stress,
                           "Stress_Level_Self_Report":Self,
                           "Physical_Activity":[0 if Physical=="High" else ( 2 if Physical=="Moderate" else 1)],
                           "Sleep_Quality":[1 if Physical=="Moderate" else ( 2 if Physical=="Poor" else 1)],
                           "Mood":[0 if Mood=="Happy" else ( 2 if Mood=="Stressed" else 1)],
                           "Work_Hour":Work})

if st.button("Predict"):
    result = rf.predict(Input_Data)
    match result:
        case 0 : st.write("High")
        case 1 : st.write("Low")
        case 2 : st.write("Moderate")
