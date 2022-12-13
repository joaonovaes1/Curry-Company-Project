# libraries 
import pandas as pd
import plotly.express as px
import numpy as np

# bibliotecas necessárias
import folium as fo
from haversine import haversine
import streamlit as st
from PIL import Image
from streamlit_folium import folium_static
import plotly.graph_objects as go

# import dataset
df = pd.read_csv('dataset/train.csv')

# ===========================================================
# Funções
# ===========================================================
def avg_std_time_graph(df1):
    df_aux = (df1.loc[:, ['Time_taken(min)', 'City']].groupby('City')
                                                    .agg({'Time_taken(min)': ['mean', 'std']})
                                                    .reset_index())
            
    df_aux.columns = ['City', 'media', 'desvio_padrao']
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Control', x=df_aux['City'], y=df_aux['media'], error_y=dict(type='data', array=df_aux['desvio_padrao'])))
    fig.update_layout(barmode='group')
    return fig

def avg_st_time_delivery(df1, festival, op='avg_time'):
    """ Está função calcula o tempo médio e o desvio padrão do tempo de entrega.
                
        Parâmetros: 
        Input:
            - df: Dataframe com os dados necessários para o cálculo
            - op: tipo de operação que precisa ser calculado
            'avg_time': calcula o tempo médio
            'std_time': calcula o desvio padrão do tempo.
        Output:
            - df: DataFrame com 2 colunas e 1 linha.

    """
    df_aux = (df1.loc[:, ['Time_taken(min)', 'Festival']]
                             .groupby('Festival')
                             .agg({'Time_taken(min)': ['mean', 'std']}))
                
    df_aux.columns = ['avg_time', 'std_time']
    df_aux = df_aux.reset_index()
    df_aux = df_aux.loc[df_aux['Festival'] == festival, op]
    df_aux = np.round(df_aux, 2)
    return df_aux
                
    
def distance(df1):
    cols = ['Restaurant_latitude', 'Restaurant_longitude',
                        'Delivery_location_latitude', 'Delivery_location_longitude']
    df1['distance'] = (df1.loc[:, cols].apply(lambda x: haversine((x['Restaurant_latitude'],
                                                                    x['Restaurant_longitude']),
                                                                    (x['Delivery_location_latitude'],
                                                                    x['Delivery_location_longitude'])),
                                                                    axis=1))
    avg_distance = np.round(df1['distance'].mean(), 2)
    return avg_distance


def clean_code(df1):
    # Excluindo dados vazios 'NaN ' da coluna 'Delivery_person_Age'
    df1 = df1.loc[df1['Delivery_person_Age'] != 'NaN ', :]
    # Transformando em int
    df1['Delivery_person_Age'] = df1['Delivery_person_Age'].astype(int)
    # Transformando Delivery_person_Ratings em float
    df1['Delivery_person_Ratings'] = df1['Delivery_person_Ratings'].astype(float)
    # Excluindo dados vazios 'NaN' da coluna 'multiple_deliveries'
    df1 = df1.loc[df1['multiple_deliveries'] != 'NaN ', :]
    # Transformando multiple_deliveries para int
    df1['multiple_deliveries'] = df1['multiple_deliveries'].astype(int)
    # Transformar a coluna Order_Date em datetime64[ns]
    df1['Order_Date'] = pd.to_datetime(df1['Order_Date'], format='%d-%m-%Y')
    # Excluindo dados vazios 'NaN ' da coluna 'Road_traffic_density'
    df1 = df1.loc[df1['Road_traffic_density'] != 'NaN ',:]\
    # Excluindo dados vazios 'NaN ' da coluna 'City'
    df1 = df1.loc[df1['City'] != 'NaN ', :]
    # Removendo os espaços dentro de strings/texto/object
    df1.loc[:, 'ID'] = df1.loc[:, 'ID'].str.strip()
    df1.loc[:, 'Road_traffic_density'] = df1.loc[:, 'Road_traffic_density'].str.strip()
    df1.loc[:, 'Type_of_order'] = df1.loc[:, 'Type_of_order'].str.strip()
    df1.loc[:, 'Type_of_vehicle'] = df1.loc[:, 'Type_of_vehicle'].str.strip()
    df1.loc[:, 'City'] = df1.loc[:, 'City'].str.strip()
    # Limpando a coluna 'Time_taken(min)'
    df1['Time_taken(min)'] = df1['Time_taken(min)'].apply(lambda x: x.split('(min) ')[1])
    df1['Time_taken(min)'] = df1['Time_taken(min)'].astype(int)
    return df1


# -------------------------------- Início da Estrutura Lógica do Código --------------------------------------

# import dataset
df = pd.read_csv('dataset/train.csv')

# cleaning dataset
df1 = clean_code(df)


# ===========================================================
# Barra Lateral Streamlit 
# ===========================================================


st.header('Marketplace Visão - Restaurantes')
st.sidebar.markdown('# Curry Company')

#image_path = '/Users/joaonovaes1/Documents/repos/Fast Track Course - FTC/codigo_jupyter/logo.png'
image = Image.open('logo.png')
st.sidebar.image(image, width=120)

st.sidebar.markdown('## Fastest Delivery in Town')
st.sidebar.markdown("""---""")

st.sidebar.markdown('## Selecione uma data limite')

date_slider = st.sidebar.slider(
        label='Qual data você deseja visualizar?',
        value=pd.datetime(2022, 4, 13),
        min_value=pd.datetime(2022, 2, 11),
        max_value=pd.datetime(2022, 6, 11),
        format='DD-MM-YYYY')

st.sidebar.markdown("""---""")

data_multiselect = st.sidebar.multiselect(
                    label='Quais as condições de trânsito?',
                    options=['Low', 'Medium', 'High', 'Jam'],
                    default='Low')

data_multiselect_weather = st.sidebar.multiselect(
                    label='Quais as condições climáticas?',
                    options=['conditions Sunny', 'conditions Fog', 'conditions Cloudy', 'conditions Windy', 'conditions Stormy', 'conditions Sandstorms'],
                    default='conditions Sunny')

data_multiselect_city = st.sidebar.multiselect(
                    label='Quais as cidades?',
                    options=['Urban', 'Semi-Urban', 'Metropolitian'],
                    default='Urban')


st.sidebar.markdown("""---""")
st.sidebar.markdown('#### Powered by JoãoN.')

# Filtro de data
linhas_selecionadas = df1['Order_Date'] < date_slider
df1 = df1.loc[linhas_selecionadas, :]

# Filtro de trânsito
linhas_selecionadas = df1['Road_traffic_density'].isin(data_multiselect)
df1 = df1.loc[linhas_selecionadas, :]

# Filtro de clima
linhas_selecionadas = df1['Weatherconditions'].isin(data_multiselect_weather)
df1 = df1.loc[linhas_selecionadas, :]

# Filtro de cidade
linhas_selecionadas = df1['City'].isin(data_multiselect_city)


# ===========================================================
# Layout Streamlit
# ===========================================================


tab1, tab2, tab3 = st.tabs(['Visão Gerencial', 'Dashboard 02', 'Dashboard 03'])

with tab1:
    with st.container():
        st.title('Overall Metrics')
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            delivery_unique = len(df1.loc[:, 'Delivery_person_ID'].unique())
            col1.metric('Entregadores', delivery_unique)
            
        with col2:
            avg_distance = distance(df1)
            col2.metric('A distância média', avg_distance)
           
        with col3:
            df_aux = avg_st_time_delivery(df1, 'Yes ', 'avg_time')
            col3.metric('Tempo médio de entrega c/ Festival', df_aux)
   
        with col4:
            df_aux = avg_st_time_delivery(df1, 'Yes ', 'std_time')
            col4.metric('Desvio padrão das entregas c/ Festival', df_aux)
    
        with col5:
            df_aux = avg_st_time_delivery(df1, 'No ', 'avg_time')
            col5.metric('Tempo de entrega médio s/ Festival', df_aux)
            
        with col6:
            df_aux = avg_st_time_delivery(df1, 'No ', 'std_time')
            col6.metric('Desvio padrão das entregas s/ Festival', df_aux)
    
    with st.container():
        st.title('Distribuição da distância')
        cols = ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']
        df1['distance'] = df1.loc[:, cols].apply(lambda x: 
                                                 haversine((x['Restaurant_latitude'], 
                                                            x['Restaurant_longitude']), 
                                                           (x['Delivery_location_latitude'], 
                                                            x['Delivery_location_longitude'])), axis=1)


        avg_distance = df1.loc[:, ['City', 'distance']].groupby('City').mean().reset_index()


        fig = go.Figure(data=[go.Pie(labels=avg_distance['City'], values=avg_distance['distance'], pull=[0, 0.1, 0])])
        st.plotly_chart(fig)

    with st.container():
        st.title('Distribuição do tempo')
        col1, col2 = st.columns(2, gap='large')
        with col1:
            fig = avg_std_time_graph(df1)
            st.plotly_chart(fig)
    
        with col2:
            df_aux = df1.loc[:, ['Time_taken(min)','City', 'Road_traffic_density']].groupby(['City',     'Road_traffic_density']).agg({'Time_taken(min)': ['mean', 'std']})
            
            df_aux.columns = ['media', 'desvio_padrao']
            df_aux = df_aux.reset_index()

            fig = px.sunburst(df_aux, path=['City', 'Road_traffic_density'], values='media', 
                             color='desvio_padrao', color_continuous_scale='RdBu', 
                             color_continuous_midpoint=np.average(df_aux['desvio_padrao']))
            
            st.plotly_chart(fig)
            
    with st.container():
        st.title('Tempo médio de entrega por cidade')
        st.markdown('''---''')
        