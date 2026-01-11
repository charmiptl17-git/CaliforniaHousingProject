# front end lib : streamlit, Flask , django,jsreact
import streamlit as st
import pandas as pd
import pickle

st.title("""Housing Price Prediction app which we design using machine learning and for the predictions of the price of the house design by XYZ Brokerage company""")

lr_mod=pickle.load(open('model_lr.pkl','rb'))
gb_mod=pickle.load(open('model_gb.pkl','rb'))

l=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
        'ocean_proximity']

input_data={}
for i in l:
   input_data[i]=st.number_input(f'Enter the value for {i}',key=i,value=0.0)

unknown_data=pd.DataFrame(input_data,index=[0])

st.subheader("Select the model you want to use for prediction")
model_option=st.selectbox('Select the model', ('Linear Regression', 'Gradient Boosting'))
if model_option=='Linear Regression':
    choose_model=lr_mod
else: 
    choose_model=gb_mod
       
if st.button('Predict the price'):
    result=choose_model.predict(unknown_data)

    st.subheader(f'The predicted price of the house is: ${result[0]}')
