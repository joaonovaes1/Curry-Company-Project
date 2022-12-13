# libraries 
import pandas as pd
import plotly.express as px

# bibliotecas necessárias
import folium as fo
from haversine import haversine
import streamlit as st
from PIL import Image
from streamlit_folium import folium_static


# import dataset
df = pd.read_csv('dataset/train.csv')

# ===========================================================
# Funções
# ===========================================================
def country_maps(df1):
            # Agrupar as cidades por cidade e tipo de tráfego 
            cols = ['Road_traffic_density', 'City', 'Delivery_location_latitude', 'Delivery_location_longitude']
            df_aux = (df1.loc[:, cols].groupby(['City', 'Road_traffic_density'])
                                      .median()
                                      .reset_index())
            # Plotar mapa
            map_ = fo.Map()
            for index, location_info in df_aux.iterrows():
                fo.Marker([location_info['Delivery_location_latitude'],
                location_info['Delivery_location_longitude']],
                popup=location_info[['City', 'Road_traffic_density']]).add_to(map_)
            folium_static(map_, width=1024, height=600)    
            return None 

def order_share_week(df1):
            df1['week_of_year'] = df1['Order_Date'].dt.strftime('%U')
            df_aux01 = df1.loc[:, ['ID', 'week_of_year']].groupby('week_of_year').count().reset_index()
            df_aux02 = (df1.loc[:, ['Delivery_person_ID', 'week_of_year']]
                           .groupby('week_of_year')
                           .nunique()
                           .reset_index())
            df_aux = pd.merge(df_aux01, df_aux02, how='inner')
            df_aux['order_by_deliver'] = df_aux['ID'] / df_aux['Delivery_person_ID']
            fig = px.line(df_aux, x='week_of_year', y='order_by_deliver')
            return fig
        
def order_by_week(df1):
    df1['week_of_year'] = df1['Order_Date'].dt.strftime('%U')
    df_aux = df1.loc[:, ['ID', 'week_of_year']].groupby('week_of_year').count().reset_index()
    # Plotar gráfico de linha
    fig = px.line(df_aux, x='week_of_year', y='ID')
    return fig


def traffic_order_city(df1):
    df_aux = (df1.loc[:, ['ID', 'City', 'Road_traffic_density']]
                            .groupby(['City',       'Road_traffic_density'])
                            .count()
                            .reset_index())
    fig = px.scatter(df_aux, x='City', y='Road_traffic_density', size='ID', color='City')
    return fig


def traffic_order_share(df1):
        # Agrupar os pedidos por tipo de tráfego
        df_aux = (df1.loc[:, ['ID', 'Road_traffic_density']]
                  .groupby('Road_traffic_density')
                  .count()
                  .reset_index())
        df_aux['entregas_percent'] = df_aux['ID'] / df_aux['ID'].sum()
        # Gráfico de Pizza
        fig = px.pie(df_aux, values='entregas_percent', names='Road_traffic_density')   
        return fig

    
def order_metric(df1):
    # Seleção de linhas
    df_aux = df1.loc[:, ['ID', 'Order_Date']].groupby('Order_Date').count().reset_index()
    # Desenha o gráfico de linhas
    fig = px.bar(df_aux, x='Order_Date', y='ID')
    return fig


def clean_code(df1):
    """ Está função tem a responsabilidade de limpar o dataframe
    
    Tipos de limpeza:
    1. Remoção dos dados NaN
    2. Mudança do tipo da coluna de dados
    3. Remoção dos espaços das variáveis de texto
    4. Formatação da data
    5. Limpeza da coluna de tempo (remoção do texto da variável númerica)
    
    Input: DataFrame
    Output: DataFrame
    """
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

# copy
df1 = clean_code(df)

# ===========================================================
# Barra Lateral Streamlit 
# ===========================================================


st.header('Marketplace Visão - Cliente')
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
#st.header(date_multiselect) --> não é necessário

st.sidebar.markdown("""---""")
st.sidebar.markdown('#### Powered by JoãoN.')

# Filtro de Data
linhas_selecionadas = df1['Order_Date'] < date_slider
df1 = df1.loc[linhas_selecionadas, :]

# Filtro de trânsito
linhas_selecionadas = df1['Road_traffic_density'].isin(data_multiselect)
df1 = df1.loc[linhas_selecionadas, :]

# ===========================================================
# Layout Streamlit
# ===========================================================


tab1, tab2, tab3 = st.tabs(['Visão Gerencial', 'Visão Tática', 'Visão Geográfica'])

with tab1:
    
    with st.container():
        # Order Metrics
        fig = order_metric(df1)
        st.markdown('# Order by Day')
        st.plotly_chart(fig, use_container_width=True)
        
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            fig = traffic_order_share(df1)
            st.markdown('#### Orders by Traffic')
            st.plotly_chart(fig, use_container_width=True)
               
        with col2: 
            st.markdown('#### Traffic Order City')
            fig = traffic_order_city(df1)
            st.plotly_chart(fig, use_container_width=True)
        
with tab2:
    with st.container():
        st.markdown('# Order by Week')
        fig = order_share_week(df1)
        st.plotly_chart(fig, use_container_width=True)
    
    with st.container():
        st.markdown('# Order Share by Week')
        fig = order_share_week(df1)
        st.plotly_chart(fig, use_container_width=True)
        

with tab3:
    with st.container():
        st.markdown('Country Map')
        country_maps(df1)



