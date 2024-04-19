
import streamlit as st 
import altair as alt
import pickle
import matplotlib.pyplot as plt 
from statsmodels.graphics.tsaplots import plot_predict 
from statsmodels.tsa.arima_model import ARIMA
with open('arima_model.pkl','rb') as file:
    model_fit = pickle.load(file)
st.set_page_config(layout="wide")
st.title("Prediction of mamber of Breakdowns with ARIMA Mode1")


st.header("How many days you want to forecast?")
p = st.number_input("days", min_value=0, max_value=365, step=1)
col1, col2 = st.columns(2)
# Generate predictions based on user input
if st.button ("Generate Predictions"):
#Perform prediction using the ARIMA model
#fig, axes = plt. subplots()
    with col1:
        st.pyplot (plot_predict (model_fit, 1, 3290+p))
    with col2:
        st.header("Predictions")
        if p==0:
            st.write('Zero Predictions')
        else:
           forecasts = model_fit.forecast(p)
           st.write(forecasts)
