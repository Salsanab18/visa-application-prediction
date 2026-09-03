import streamlit as st
import eda
import prediction

page = st.sidebar.selectbox('Tampilkan Page: ', ('EDA', 'Prediction'))

if page == 'EDA':
    eda.run()
else:
    prediction.run()