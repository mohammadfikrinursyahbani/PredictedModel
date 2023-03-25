# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import joblib

# st.write("""
# # Boston House Price Prediction App
# This app predicts the **Boston House Price**!
# """)
# st.write('---')

# st.sidebar.header('Specify Input Parameters')

# X = pd.read_csv('boston_housing.csv')


# def user_input_features():
#     crim = st.sidebar.slider('CRIM', float(X.crim.min()), float(
#         X.crim.max()), float(X.crim.mean()))
#     zn = st.sidebar.slider('ZN', float(X.zn.min()),
#                            float(X.zn.max()), float(X.zn.mean()))
#     indus = st.sidebar.slider('INDUS', float(X.indus.min()), float(
#         X.indus.max()), float(X.indus.mean()))
#     chas = st.sidebar.slider('CHAS', float(X.chas.min()), float(
#         X.chas.max()), float(X.chas.mean()))
#     nox = st.sidebar.slider('NOX', float(X.nox.min()),
#                             float(X.nox.max()), float(X.nox.mean()))
#     rm = st.sidebar.slider('RM', float(X.rm.min()),
#                            float(X.rm.max()), float(X.rm.mean()))
#     age = st.sidebar.slider('AGE', float(X.age.min()),
#                             float(X.age.max()), float(X.age.mean()))
#     dis = st.sidebar.slider('DIS', float(X.dis.min()),
#                             float(X.dis.max()), float(X.dis.mean()))
#     rad = st.sidebar.slider('RAD', float(X.rad.min()),
#                             float(X.rad.max()), float(X.rad.mean()))
#     tax = st.sidebar.slider('TAX', float(X.tax.min()),
#                             float(X.tax.max()), float(X.tax.mean()))
#     ptratio = st.sidebar.slider('PTRATIO', float(X.ptratio.min()), float(
#         X.ptratio.max()), float(X.ptratio.mean()))
#     b = st.sidebar.slider('B', float(X.b.min()),
#                           float(X.b.max()), float(X.b.mean()))
#     lstat = st.sidebar.slider('LSTAT', float(X.lstat.min()), float(
#         X.lstat.max()), float(X.lstat.mean()))
#     data = {'CRIM': crim,
#             'ZN': zn,
#             'INDUS': indus,
#             'CHAS': chas,
#             'NOX': nox,
#             'RM': rm,
#             'AGE': age,
#             'DIS': dis,
#             'RAD': rad,
#             'TAX': tax,
#             'PTRATIO': ptratio,
#             'B': b,
#             'LSTAT': lstat}
#     features = pd.DataFrame(data, index=[0])
#     return features


# df = user_input_features()

# model = joblib.load('svr_model.sav')


# if st.button('Predict'):
#     prediction = model.predict(df)
#     st.header('Prediction of MEDV')
#     st.write(prediction)
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
