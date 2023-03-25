!pip install joblib
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model
model = joblib.load('xgb_model.pkl')

# Define the app title and sidebar heading
st.title('House Price Prediction App')
st.sidebar.header('Enter House Details')

# Define the input fields in the sidebar
crim = st.sidebar.number_input('crim', value=0)
zn = st.sidebar.number_input(
    'zn', value=0)
indus = st.sidebar.number_input(
    'indus', value=0)
chas = st.sidebar.selectbox('chas', [0, 1], index=0)
nox = st.sidebar.number_input(
    'nox', value=0)
rm = st.sidebar.number_input('rm', value=0)
age = st.sidebar.number_input(
    'age', value=0)
dis = st.sidebar.number_input(
    'dis', value=0)
rad = st.sidebar.number_input(
    'rad', value=0)
tax = st.sidebar.number_input(
    'tax', value=0)
ptratio = st.sidebar.number_input('ptratio', value=0)
b = st.sidebar.number_input(
    'b', value=0)
lstat = st.sidebar.number_input('lstat', value=0)

# Define the input data as a dictionary
input_data = {'CRIM': crim, 'ZN': zn, 'INDUS': indus, 'CHAS': chas,
              'NOX': nox, 'RM': rm, 'AGE': age, 'DIS': dis, 'RAD': rad,
              'TAX': tax, 'PTRATIO': ptratio, 'B': b, 'LSTAT': lstat}

# Create a dataframe from the input data
input_df = pd.DataFrame([input_data])

# Define the predicted value as the median value of owner-occupied homes in $1000's
if st.button('Predict'):
    target = '$' + str(int(np.round(model.predict(input_df)[0] * 1000)))
    # Display the predicted value
    st.subheader('Predicted House Price:')
    st.write(target)
