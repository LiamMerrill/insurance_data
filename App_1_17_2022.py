#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:30:27 2022

@author: liamsweeney
"""

import pandas as pd
import numpy as np
import plotly.express as px
#import plotly.graph_objects as go
#import category_encoders as ce
#import matplotlib.pyplot as pl
#from sklearn.pipeline import make_pipeline
#from sklearn.model_selection import train_test_split
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.tree import plot_tree
#from pdpbox import pdp, info_plots
#from sklearn.pipeline import make_pipeline
#import sklearn
import streamlit as st
import pickle

st.title("Insurance Data")
url = 'https://raw.githubusercontent.com/LiamMerrill/insurance_data/main/insurance_premiums.csv'

num_rows = st.sidebar.number_input('Select Number of Rows to Load', 
                                   min_value=1000,
                                   max_value=50000,
                                   step=1000)

section = st.sidebar.radio('Choose Application Section', ['Data Explorer','Model Explorer'])
print(section)

@st.cache
def load_data(num_rows):
    df = pd.read_csv(url, nrows = num_rows)
    return df

@st.cache
def create_grouping(x_axis, y_axis):
    grouping = df.groupby(x_axis)[y_axis].mean()
    return grouping

def load_model():
    with open('pipe.pkl', 'rb') as pickled_mod:
        model = pickle.load(pickled_mod)
    return model

df = load_data(num_rows)

if section == 'Data Explorer':
    
    
    x_axis = st.sidebar.selectbox("Choose column for X-axis", 
                                  df.select_dtypes(include = np.object).columns.tolist())
    
    y_axis = st.sidebar.selectbox("Choose column for y-axis", ['charges', 
                                                               'bmi'])
    
    chart_type = st.sidebar.selectbox("Choose Your Chart Type", 
                                      ['line', 'bar', 'area'])
    
    if chart_type == 'line':
        grouping = create_grouping(x_axis, y_axis)
        st.line_chart(grouping)
        
    elif chart_type == 'bar':
        grouping = create_grouping(x_axis, y_axis)
        st.bar_chart(grouping)
    elif chart_type == 'area':
        fig = px.strip(df[[x_axis, y_axis]], x=x_axis, y=y_axis)
        st.plotly_chart(fig)
    
    st.write(df)
    
else:
    st.text("Choose Options to the Side to Explore the Model")
    model = load_model()
    
   
    
    
    age = st.sidebar.number_input("Age")
    
    bmi = st.sidebar.number_input("Body Mass Index (BMI)", min_value = 10,
                                        max_value = 200, step = 5, value = 20)
    children = st.sidebar.number_input("children")
                            
    region = st.sidebar.number_input("region")
    
    count = st.sidebar.number_input("count")
    
    sex_female = st.sidebar.number_input("female")
    
    sex_male = st.sidebar.number_input("male")
    
    smoker_no = st.sidebar.number_input("non smoker")
                           
    smoker_yes = st.sidebar.selectbox("smoker", 
                                       df['smoker'].unique().tolist())
    
    sample = {
    'age': age,
    'bmi': bmi,
    'children': children,
    'region': region,
    'count': count,
    'sex_female': sex_female,
    'sex_male': sex_male,
    'smoker_no': smoker_no,
    'smoker_yes': smoker_yes
    
    }

    sample = pd.DataFrame(sample)
    prediction = model.predict(sample)[0]
    
    st.title("Predicted Charges")
  