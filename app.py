import pandas as pd
import plotly.express as px
import streamlit as st
import pickle as pkl
import numpy as np

gpa = pd.read_csv('gpa.csv')

st.title("Notas prom Uni")

with open("model.pickle", "rb") as m:
    modelo = pkl.load(m)

Tab1, Tab2, Tab3 = st.tabs(['Analisis Univariado', 'Analisis Bivariado', 'Modelo'])
 
with Tab1:
   st.subheader("Estadísticas descriptivas - Variables numéricas")
   st.dataframe(gpa[['prom_uni', 'prom_colegio', 'ACT', 'capar_clase']].describe())
 
   st.subheader("Distribución de variables numéricas")
 
   fig = px.histogram(gpa, x='prom_uni', title='prom universidad')
   st.plotly_chart(fig)
 
   fig2 = px.histogram(gpa, x='prom_colegio', title='prom colegio')
   st.plotly_chart(fig2)
 
   fig3 = px.histogram(gpa, x='ACT', title='prom prueba ACT')
   st.plotly_chart(fig3)
 
   fig4 = px.histogram(gpa, x='capar_clase', title='prom de faltas a clase')
   st.plotly_chart(fig4)

with Tab2:
    st.subheader("Relación entre variables y el promedio ponderado en la U")
 
    fig1 = px.box(gpa, x='ACT', y='prom_uni', title='Promedio en la Uni vs Puntaje ACT')
    st.plotly_chart(fig1)
 
    fig2 = px.box(gpa, x='prom_colegio', y='prom_uni', title='Promedio en la U vs Promedio en el colegio')
    st.plotly_chart(fig2)

    fig3 = px.box(gpa, x='capar_clase', y='prom_uni', title='Promedio en la U vs Cant. veces que faltó a clase')
    st.plotly_chart(fig3)
 

with Tab3:
    st.title("Modelo")
 
    prom_colegio = st.slider("Rendimiento en el colegio", 1, 5)
 
    ACT = st.slider("Prueba ACT", 1, 36)
 
    capar_clase= st.slider("Falto a clases en la semana", 0, 8)
 
    if st.button ("Predecir"):
        Predecir = modelo.predict(np.array([[prom_colegio, ACT, capar_clase]]))
        st.write(f"El promedio es: {Predecir [0]}")
 
print(10)
