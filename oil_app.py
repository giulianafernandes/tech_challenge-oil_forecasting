#bibliotecas
import streamlit as st
import pandas as pd
import pickle
import matplotlib as plt
import plotly.graph_objects as go
import numpy as np
import plotly.graph_objects as go

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# from keras.optimizers import Adam
from keras.metrics import MeanSquaredError

from sklearn.preprocessing import MinMaxScaler

import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

# >>>>>>>>> HEADER

st.header(':red[TECH CHALLENGE:] Oil Forecasting', divider = 'rainbow')

st.markdown('''Você foi contratado(a) para uma consultoria, e seu trabalho envolve analisar os dados de preço do petróleo **Brent**, que pode ser encontrado no site do Ipea. 
            Essa base de dados histórica envolve duas colunas: data e preço (em dólares). :fuelpump:''')

st.markdown('''Um grande cliente do segmento pediu para que a consultoria desenvolvesse um **dashboard interativo** e que gere insights relevantes para tomada de decisão. 
            Além disso, solicitaram que fosse desenvolvido um **modelo de Machine Learning** para fazer o forecasting do preço do petróleo.''')


# carregando o dataframe
url = ('https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls')
dados = pd.read_excel(url, sheet_name="Data 1", skiprows=2)
dados = dados.rename(columns={'Date': 'date',
                        'Europe Brent Spot Price FOB (Dollars per Barrel)': 'dollars_per_barrel'})

#>>>>>>>>> GRÁFICO
st.image('images/Imagem1-1.png')

st.subheader('Preço do Petróleo BRENT desde 1987', divider = 'orange')

st.line_chart(data= dados, x='date', y='dollars_per_barrel', 
              color='#df4d4d', use_container_width=True)


#>>>>>>>>> TABS

tab1, tab2, tab3, tab4 = st.tabs(["Modelo", "Dashboard", "Contexto Histórico", "Info"])

with tab1:
    #Modelo de Machine Learning
    # /////////////////////////////////////////
    
    #carregando o modelo
    with open('modelo_brent.pkl', 'rb') as file_2:
        modelo_brent = pickle.load(file_2)


    #>>>>>> começo 
    dados_lstm =  dados.loc['2014-05-01':]
    #dados_lstm = dados_lstm.reset_index('date')

    price_data = dados_lstm['dollars_per_barrel'].values
    price_data = price_data.reshape(-1,1) #transformando em array

    # normalizando os dados
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler = scaler.fit(price_data)
    price_data = scaler.transform(price_data)

    # sepando em treino e teste
    porcentagem = 0.80
    split = int(porcentagem*len(price_data))

    price_train = price_data[:split]
    price_test = price_data[split:]

    date_train = dados_lstm['date'][:split]
    date_test = dados_lstm['date'][split:]

    # gera sequências temporais para treinamento e teste em um modelo de aprendizado de máquina
    look_back = 10
    train_generator = TimeseriesGenerator(price_train, price_train, length=look_back, batch_size=20)
    test_generator = TimeseriesGenerator(price_test, price_test, length=look_back, batch_size=1)



    # carregando o modelo
    with open('modelo_brent.pkl', 'rb') as file_2:
        modelo_brent = pickle.load(file_2)

    # previsoes usando o conjunto de teste
    test_predictions = modelo_brent.predict(test_generator)

    # inverte qualquer transformação aplicada nos dados
    test_predictions_inv = scaler.inverse_transform(test_predictions.reshape(-1,1))
    test_actual_inv = scaler.inverse_transform(np.array(price_test).reshape(-1,1))

    #ajusta as dimensões
    test_actual_inv = test_actual_inv[:len(test_predictions_inv)]

    # métricas
    mse = modelo_brent.evaluate(test_generator, verbose=1)
    mape = np.mean(np.abs((test_actual_inv - test_predictions_inv) / test_actual_inv)) * 100
    rmse = np.sqrt(mse[0]) #rmse é a raiz quadrada do mse - média dos quadrados das diferenças entre as previsões a os dados reais

    prediction = modelo_brent.predict(test_generator)

    price_train = price_train.reshape((-1))
    price_test = price_test.reshape((-1))
    prediction = prediction.reshape((-1))


    st.subheader('Predições do Preço do Barril de Petróleo BRENT', divider = 'orange')

    # métricas
    st.markdown(f'#### 📈 mape: {mape:.4f} | rmse: {rmse:.4f}')
    
    # >>>>>>> tabela
    def predict(num_prediction, model):
        prediction_list = price_data[-look_back:]
        
        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back-1:]
        
        return prediction_list

    def predict_dates (num_prediction):
        last_date = dados_lstm['date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
        return prediction_dates

    num_prediction = 15
    forecast = predict(num_prediction, modelo_brent)
    forecast_dates = predict_dates(num_prediction)
    df_past = dados.copy()
    df_past['date'] = pd.to_datetime(df_past['date'])
    df_past['forecast'] = np.nan
    df_past['forecast'].iloc[-1] = df_past['dollars_per_barrel'].iloc[-1]
    forecast = forecast.reshape(-1,1)
    forecast = scaler.inverse_transform(forecast)
    df_future = pd.DataFrame(columns=['date', 'dollars_per_barrel', 'forecast'])
    df_future['date'] = forecast_dates
    df_future['forecast'] = forecast.flatten()
    df_future['dollars_per_barrel'] = np.nan
    frames = [df_past, df_future]
    results = pd.concat(frames, ignore_index=True).set_index('date')
    results = results.rename(columns={'date': 'Data', 
                                    'dollars_per_barrel': 'Preço real | US$ por Barril',
                                    'forecast': 'Previsão'})

    st.dataframe(results.tail(21), width=1000)
    
    # >>>>>>> gráfico
    
    results_lastyear = results.loc['2023-05-01':]
    
    plot_data = [go.Scatter(x= results_lastyear.index,
                            y= results_lastyear['Preço real | US$ por Barril'],
                            name= 'Valor Real'),
                go.Scatter(x= results_lastyear.index,
                            y= results_lastyear['Previsão'],
                            name= 'Predição')]

    plot_layout = go.Layout(title= 'Forecast - Barril de Petróleo Brent')
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig, use_container_width=True)
    
    # /////////////////////////////////////////
    st.image('images/Imagem5-5.png')

with tab2:
    #Dashboard Power Bi entra aqui 
    st.subheader('Dashboard', divider = 'orange')
    st.image('images/dashboard.jpeg')
    st.image('images/dashboard2.jpeg')
    st.image('images/Imagem6-6.png')

with tab3:
    #Situações geopoliticas que podem ter impactado em certos momentos na demanda pelo barril
    st.subheader('Contexto Histórico', divider = 'orange')
    st.image('images/Imagem2-2.png')
    
with tab4:
    #Situações geopoliticas que podem ter impactado em certos momentos na demanda pelo barril
    st.subheader('Informações sobre o trabalho', divider = 'orange')
    st.markdown('**GRUPO 41: Giuliana de Sousa Fernandes RM352002 e Larissa Suelen Lima da Silva RM350906**')
    st.markdown('Dados adquiridos no site da EIA - U.S. Energy Information Administration')
    
    st.link_button('Europe Brent Spot Price', 'https://www.eia.gov/dnav/pet/hist/rbrteD.htm')
    
    st.link_button('GitHub do projeto', 'https://github.com/giulianafernandes/tech_challenge-oil_forecasting')
    
    st.image('images/Imagem3.png')