# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:02:43 2022

@author: Kevin Boss
"""
from PIL import Image
#from streamlit_shap import st_shap
import streamlit as st
import pandas as pd 
import time
import plotly.express as px 
import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,roc_auc_score
import shap
#import catboost
#from catboost import CatBoostClassifier
import pickle
import xgboost as xgb
from xgboost import XGBClassifier
# load the saved model
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import numpy as np 

def xgb_shap_transform_scale(original_shap_values, Y_pred, which):    
    from scipy.special import expit    
    #Compute the transformed base value, which consists in applying the logit function to the base value    
    from scipy.special import expit 
    #Importing the logit function for the base value transformation    
    untransformed_base_value = original_shap_values.base_values[-1]    
    #Computing the original_explanation_distance to construct the distance_coefficient later on    
    original_explanation_distance = np.sum(original_shap_values.values, axis=1)[which]    
    base_value = expit(untransformed_base_value) 
    # = 1 / (1+ np.exp(-untransformed_base_value))    
    #Computing the distance between the model_prediction and the transformed base_value    
    distance_to_explain = Y_pred[which] - base_value    
    #The distance_coefficient is the ratio between both distances which will be used later on    
    distance_coefficient = original_explanation_distance / distance_to_explain    
    #Transforming the original shapley values to the new scale    
    shap_values_transformed = original_shap_values / distance_coefficient    
    #Finally resetting the base_value as it does not need to be transformed    
    shap_values_transformed.base_values = base_value    
    shap_values_transformed.data = original_shap_values.data    
    #Now returning the transformed array    
    return shap_values_transformed


plt.style.use('default')

st.set_page_config(
    page_title = 'Real-Time Fraud Detection',
    page_icon = '🕵️‍♀️',
    layout = 'wide'
)

# dashboard title
#st.title("Real-Time Fraud Detection Dashboard")
#st.markdown("<h1 style='text-align: center; color: black;'>机器学习： 实时识别出虚假销售</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>A Prediction Model for </h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Cirrhotic patients with AKI (PMC-AKI)</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'> </h1>", unsafe_allow_html=True)


# side-bar 
def user_input_features():
    st.sidebar.header('Make a prediction')
    st.sidebar.write('User input parameters below ⬇️')
    #a0 = st.sidebar.slider('Hcy (umol/L)', 0.0, 100.0, 0.0)
    a1 = st.sidebar.slider('Sex (1=male, 0=female)', 0, 1, 1)
    a2 = st.sidebar.slider('Total bilirubin (μmol/L)', 0.0, 300.0, 0.0)
    a3 = st.sidebar.slider('Albumin (g/L)', 0.5, 100.0, 0.0)
    a4 = st.sidebar.slider('Serum creatinine (μmol/L)', 0.0, 500.0, 0.0)
    a5 = st.sidebar.slider('INR', 0.0, 6.0, 0.0)
    a6 = st.sidebar.slider('Bile acid (μmol/L)', 1.0, 100.0, 0.0)
    a7 = st.sidebar.slider('WBC (10^9/L)', 1.0, 20.0, 0.0)
    a8 = st.sidebar.slider('ESR (mm/h)', 0.0, 100.0, 0.0)


    output = [a1,a2,a3,a4,a5,a6,a7,a8]
    return output

outputdf = user_input_features()


#online delete position
with open('XGB20231227_.pkl', 'rb') as f:
    catmodel = pickle.load(f)

outputdf_ = outputdf

#st.header('👉 Make predictions in real time')
colnames = ['Sex','Total bilirubin','Albumin','Serum creatinine','INR',
                     'Bile acid','WBC','ESR']
outputdf = pd.DataFrame([outputdf], columns= colnames)

colnames_ = ['Sex (1=male, 0=female)','Total bilirubin (μmol/L)','Albumin (g/L)','Serum creatinine (μmol/L)',
'INR', 'Bile acid (μmol/L)','WBC (10^9/L)','ESR (mm/h)']
outputdf_ = pd.DataFrame([outputdf_], columns= colnames_)

#st.write('User input parameters below ⬇️')
#st.write(outputdf)


#p1 = catmodel.predict(dtest)[0]
#p2 = catmodel.predict_proba(dtest)
p2 = catmodel.predict_proba(outputdf)[:, 1]
p1 = 0
if p2 < 0.1:
    p1 = 'Low risk of AKI'
elif p2 >= 0.4:
    p1 = 'High risk of AKI'
else:
    p1 = 'Medium risk of AKI'
#p1 = (p2 >= 0.232816)*1#best threshold

#modify output dataframe
outputdf_1 = outputdf_.iloc[:,0:4]
outputdf_2 = outputdf_.iloc[:,4:8]

placeholder6 = st.empty()
with placeholder6.container():
    st.subheader('Part1: User input parameters below ⬇️')
    st.write(outputdf_1)
    st.write(outputdf_2)


placeholder7 = st.empty()
with placeholder7.container():
    st.subheader('Part2: Output results ⬇️')
    st.write(f'1. Predicted class: {p1}')
    st.write(f'2. Predicted class probability: {p2}')
   

