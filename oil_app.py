import streamlit as st
import pandas as pd

st.header(':red[TECH CHALLENGE:] Oil Forecasting', divider = 'rainbow')

st.markdown('''Você foi contratado(a) para uma consultoria, e seu trabalho envolve analisar os dados de preço do petróleo **Brent**, que pode ser encontrado no site do Ipea. 
            Essa base de dados histórica envolve duas colunas: data e preço (em dólares). :fuelpump:''')

st.markdown('''Um grande cliente do segmento pediu para que a consultoria desenvolvesse um **dashboard interativo** e que gere insights relevantes para tomada de decisão. 
            Além disso, solicitaram que fosse desenvolvido um **modelo de Machine Learning** para fazer o forecasting do preço do petróleo.''')

df = pd.read_excel('data/RBRTEd.xls',sheet_name="Data 1")

st.image('images/Imagem1-1.png')

st.subheader('Preço do Petróleo BRENT desde 1987', divider = 'orange')

st.line_chart(data= df, x='Date', y='Europe Brent Spot Price FOB (Dollars per Barrel)', 
              color='#df4d4d', use_container_width=True)


tab1, tab2, tab3 = st.tabs(["Dashboard", "Contexto Histórico", "Modelo"])

with tab1:
    #Dashboard Power Bi entra aqui 
    st.subheader('Dashboard', divider = 'orange')
    st.image('images/Imagem6-6.png')

with tab2:
    #Situações geopolóticas que podem ter impactado em certos momentos na demanda pelo barril
    st.subheader('Contexto Histórico', divider = 'orange')
    st.image('images/Imagem2-2.png')

with tab3:
    #Modelo de Machine Learning
    st.subheader('Modelo de ML', divider = 'orange')
    st.image('images/Imagem5-5.png')