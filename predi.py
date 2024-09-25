#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# Updated data for confusion matrices
confusion_matrices = {
    'OLS': [
        [-1.0, 0.0, 1.0, 2.0, 3.0, 197142903.0],
        [-1.0, 0, 0, 0, 0, 0],
        [0.0, 5, 1137, 170, 1, 0],
        [1.0, 1, 10, 36, 2, 1],
        [2.0, 0, 0, 0, 0, 0],
        [3.0, 0, 0, 0, 0, 0],
        [197142903.0, 0, 0, 0, 0, 0]
    ],
    'Ridge': [
        [0, 0, 0, 0, 0, 0],
        [5, 1129, 179, 0, 0, 0],
        [0, 7, 41, 1, 1, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ],
    'Lasso': [
        [826, 487, 0],
        [20, 30, 1],
        [0, 0, 0]
    ],
    'RFE Logistic Regression': [
        [726, 4],
        [587, 47]
    ],
    'Polynomial Logistic Regression': [
        [1106, 207],
        [30, 21]
    ]
}

# Updated metrics based on your classification reports
metrics = {
    'Model': ['OLS', 'Ridge', 'Lasso', 'RFE Logistic Regression', 'Polynomial Logistic'],
    'Accuracy': [0.86, 0.86, 0.63, 0.57, 0.83],
    'Recall': [0.86, 0.86, 0.63, 0.57, 0.83],
    'Precision': [0.96, 0.96, 0.94, 0.96, 0.94],
    'F1 Score': [0.90, 0.90, 0.74, 0.69, 0.88]
}
df_metrics = pd.DataFrame(metrics)

# Function to display confusion matrices as tables
def display_confusion_matrix_table(cm, model_name):
    try:
        cm_df = pd.DataFrame(cm)
        st.write(f'Confusion Matrix for {model_name}')
        st.table(cm_df)
    except Exception as e:
        st.error(f"Error displaying confusion matrix for {model_name}: {e}")

# Streamlit app layout
st.title('Bankruptcy Prediction Model Comparison')

# Display confusion matrices as tables
for model_name, cm in confusion_matrices.items():
    display_confusion_matrix_table(cm, model_name)


# In[ ]:




