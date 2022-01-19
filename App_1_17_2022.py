#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 11:30:27 2022

@author: liamsweeney
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import category_encoders as ce
#import matplotlib.pyplot as pl
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from pdpbox import pdp, info_plots
from sklearn.pipeline import make_pipeline
import sklearn
import streamlit as st
import pickle

st.title("Insurance Data")
url = '/Users/liamsweeney/dat-11-15/App/Insurance Data/insurance_premiums.csv'

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
    
    id_val = st.sidebar.selectbox("Choose age", 
                                  df['age'].unique().tolist())
    #yesterday = st.sidebar.number_input("How many visitors yesterday", min_value = 0,
        #                                max_value = 100, step = 1, value = 20)
    #day_of_week = st.sidebar.selectbox("Day of Week", 
      #                                 df['day_of_week'].unique().tolist())
    
   # sample = {
    #'id': id_val,
    #'yesterday': yesterday,
    #'day_of_week': day_of_week
    #}

    #sample = pd.DataFrame(sample, index = [0])
    #prediction = model.predict(sample)[0]
    
    #st.title(f"Predicted Attendance: {int(prediction)}")
    

df = pd.read_csv('https://raw.githubusercontent.com/LiamMerrill/insurance_data/insurance_premiums.csv')

categorical = ['sex', 'smoker']

df_ohe = pd.get_dummies(df, columns=categorical)

ore = ce.OrdinalEncoder(mapping = [
    {
        'col': 'region',
        'mapping': {'northeast': 1, 'northwest': 2, 'southeast': 3, 'southwest': 4}
    }
])

df_ce = ore.fit_transform(df_ohe)

fig = px.histogram(df_ce, x="bmi", title = "Body Mass Index Histogram")
fig.show()

fig = px.scatter(df, x="bmi", y="charges", title = "Body Mass Index/Charges: Scatter Plot")
fig.show()

fig = px.histogram(df, x="charges", color = "sex", nbins = 30, title = "Sex/Charges: Histogram")
fig.show()

fig = px.scatter(df, x="age", y="charges", color="smoker", size = "bmi", title = "Age/Charges: Scatter Plot - color: smoker, size: bmi")
fig.show()

fig = px.scatter(df, x="bmi", y="charges", color="smoker", size = "age", title = "Body Mass Index/Charges: Scatter Plot - color: smoker, size: age")
fig.show()

fig = px.scatter(df, x="bmi", y="age", color="smoker", size = "charges", title = 'Body Mass Index/Age: Scatter Plot - color: smoker, size: charges')
fig.show()

tree_1 = DecisionTreeRegressor(max_depth = 3)

X = df_ce.drop("charges", axis = 1)
y = df_ce["charges"]

tree_1.fit(X, y)

import matplotlib.pyplot as plt

plt.figure(figsize = (16, 10))

plot_tree(tree_1, filled = True, fontsize = 12, feature_names = X.columns);

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.15, random_state=42)

tree_1.score(X_test, y_test)

def scoring_trees_d(depth):
    tree_1 = DecisionTreeRegressor(max_depth = depth)
    tree_1.fit(X_train, y_train)
    return tree_1.score(X_test, y_test)

def scoring_trees_s(split):
    tree_1 = DecisionTreeRegressor(min_samples_split = split)
    tree_1.fit(X_train, y_train)
    return tree_1.score(X_test, y_test)

print(scoring_trees_d(1))
print(scoring_trees_d(2))
print(scoring_trees_d(3))
print(scoring_trees_d(4))
print(scoring_trees_d(5))
print(scoring_trees_d(6))
print(scoring_trees_d(7))

print(scoring_trees_s(2))
print(scoring_trees_s(3))
print(scoring_trees_s(4))
print(scoring_trees_s(5))
print(scoring_trees_s(6))

tree_2 = DecisionTreeRegressor(max_depth = 4, min_samples_split = 6)

tree_2.fit(X_train, y_train)

plt.figure(figsize = (16, 10))

plot_tree(tree_2, filled = True, fontsize = 12, feature_names = X.columns);

tree_2.score(X_train, y_train)