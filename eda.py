import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image

def run():
    st.title('Visa Application Predict for Asian Workers')

    #Tampilan gambar
    img = Image.open('visa applicant.png')
    st.image(img)

    st.write(
        'This page shows a simple exploratory analysis of work visa applications'
        'for employees from Asia with yearly prevailing wage.'
    )

    #Load data 
    data = pd.read_csv('EasyVisa.csv')

    #filter data
    data = data[
        (data['continent'] == 'Asia') &
        (data['unit_of_wage'] == 'Year')
    ].copy()

    #Load data 
    st.write('### Dataset')
    st.dataframe(data)

    st.write('Jumlah data:', data.shape[0])
    st.write('Jumlah kolom:', data.shape[1])

    #visa aplication status
    st.write('### Visa Application Status')
    fig, ax = plt.subplots(figsize=(8, 5))
    data['case_status'].value_counts().plot(
        kind='pie',
        autopct='%1.1f%%',
        ax=ax
    )
    ax.set_ylabel('')
    st.pyplot(fig)

    certified = (data['case_status'] == 'Certified').sum()
    denied = (data['case_status'] == 'Denied').sum()
    total = len(data)

    st.write(
        f'Certified: {certified} ({certified/total*100:.1f}%)|'
        f'Denied: {denied} ({denied/total*100:.1f}%)'
    )

    #visa aplication status
    st.write('### Visa Status by Education')
    fig = plt.figure(figsize=(10, 5))
    sns.countplot(data=data, x = 'education_of_employee', 
                  hue = 'case_status')
    plt.xticks(rotation=15)
    st.pyplot(fig)

    #visa aplication status
    st.write('### Visa Status by Job Experience')
    fig = plt.figure(figsize=(8, 5))
    sns.countplot(x = 'has_job_experience', 
                  hue = 'case_status',
                  data=data)

    st.pyplot(fig)
    
    #Numerical Distribution
    st.write('### Numerical Distribution')
    
    option = st.selectbox('Choose numerical column:', ('no_of_employees', 'yr_of_estab', 'prevailing_wage'))
    fig = px.histogram(data,
        x=option,
        color='case_status',
        marginal='box'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write('### Scatter Plot with Plotly')
        
    fig= px.scatter(
        data,
        x='no_of_employees',
        y='prevailing_wage',
        hover_data=['education_of_employee', 'case_status']
    )
    st.plotly_chart(fig)

if __name__ =="__main__":
    run()