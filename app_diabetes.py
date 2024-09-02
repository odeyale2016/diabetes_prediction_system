# -*- coding: utf-8 -*-
"""
Created on Sunday September 01 15:29:50 2024

@author: Alphatech
"""

import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
#loading the saved models
#1 - diabetetic model
diabetes_model = joblib.load('diabetes_model.pkl')

#2 - heart disease model
Heart_Disease_model = joblib.load('heart_disease_model.sav')

#3 - parkinson disease model
Parkinson_Disease_model = joblib.load('parkinsons_model.sav')


#sidebar for navigation
with st.sidebar:
    selected = option_menu('Machine Learning Prediction System',['Diabetes Prediction','Heart Disease Prediction','Parkinsons Prediction'],
                           icons = ['activity','heart','person'],default_index=0)
    
#diabetes Prediction page
if(selected == 'Diabetes Prediction'):
    html_temp = """
    <div style="background-color:darkred; padding:10px">
    <h2 style="color:white; text-align:center;">Machine Learning Model to predict Diabetes </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
     
    st.write("Enter the values below to predict the likelihood of diabetes:")
    
    #taking input from  user
    col1,col2,col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1)
        
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=100)
    
    with col3:
        BloodPressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=122, value=72)
    
    with col1:
       SkinThickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20)
   
    with col2:
       Insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=846, value=79)
   
    with col3:
       BMI = st.number_input('Body Mass Index (BMI)', min_value=0.0, max_value=100.0, value=32.0)
   
    with col1:
       DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
   
    with col2:
       Age = st.number_input('Age', min_value=0, max_value=120, value=33)
       
       
    # code for prediction
    # Predict button
if st.button('Predict'):
    features = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    prediction = diabetes_model.predict(features)
    probability = diabetes_model.predict_proba(features)[0][1]

    if prediction == 1:
        st.markdown(f'<p style="color:red; background-color:#000; size:20px;">The model predicts you <strong>have diabetes</strong> with a probability of {probability:.2f}.</p>', unsafe_allow_html=True)
        st.status('Model Prediction Completed')
    else:
        st.markdown(f'<p style="color:green; background-color:#000; size:20px">The model predicts you <strong>do not have diabetes</strong> with a probability of {1 - probability:.2f}.</p>', unsafe_allow_html=True)
        st.status('Model Prediction Completed')
    
     
 
# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    
    html_temp = """
    <div style="background-color:purple; padding:10px">
    <h2 style="color:white; text-align:center;">Heart Disease Prediction using Machine Learning </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = Heart_Disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
             
# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
   
    html_temp = """
    <div style="background-color:brown; padding:10px">
    <h2 style="color:white; text-align:center;">Parkinson's Disease Prediction using ML </h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = Parkinson_Disease_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)

html_temp = """
    <div style="background-color:black; padding:10px"; color:white;>
    <h5 style="color:white; text-align:center;">&copy 2024 Created by: Odeyale Kehinde Musiliudeen </h5>
    </div>
    """

st.markdown(html_temp, unsafe_allow_html=True)
        
