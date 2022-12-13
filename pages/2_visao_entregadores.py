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

def top_delivers(df1, top_asc):
    df_aux = (df1.loc[:, ['Delivery_person_ID', 'City', 'Time_taken(min)']]
                 .groupby(['City', 'Delivery_person_ID'])
                 .mean()
                 .sort_values(['Time_taken(min)', 'City'], ascending=top_asc)
                 .reset_index())
    df_aux01 = df_aux.loc[df_aux['City'] == 'Metropolitian', :].head(10)
    df_aux02 = df_aux.loc[df_aux['City'] == 'Urban', :].head(10)
    df_aux03 = df_aux.loc[df_aux['City'] == 'Semi-Urban', :].head(10) 
    df2 = pd.concat([df_aux01, df_aux02, df_aux03]).reset_index(drop=True)
    return df2

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

# cleaning dataset
df1 = clean_code(df)


# ===========================================================
# Barra Lateral Streamlit 
# ===========================================================


st.header('Marketplace Visão - Entregadores')
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


#st.header(date_multiselect) --> não é necessário

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

# ===========================================================
# Layout Streamlit
# ===========================================================

tab1, tab2, tab3 = st.tabs(['Visão Gerencial', '', ''])

with tab1:
    with st.container():
        st.title('Overall Metrics')
        
        col1, col2, col3, col4 = st.columns(4, gap='large')
        with col1:
            # Maior idade dos entregadores
            maior_idade = df1.loc[:,'Delivery_person_Age'].max()
            col1.metric('Maior idade', maior_idade)
            
            
        with col2:
            # Menor idade dos entregadores
            menor_idade = df1.loc[:,'Delivery_person_Age'].min()
            col2.metric('Menor idade', menor_idade)
            
            
        with col3:
            melhor = df1.loc[:, 'Vehicle_condition'].max()
            col3.metric('Melhor condição veic', melhor)
            
        with col4:
            pior = df1.loc[:, 'Vehicle_condition'].min()
            col4.metric('Pior condição veic', pior)
            
    with st.container():
        st.markdown('''---''')
        st.title('Avaliações')
        col1, col2 = st.columns(2, gap='large')
        
        with col1:
            st.markdown('##### Avaliação média por entregador')
            df_avg_ratings_per_deliver = (df1.loc[:, ['Delivery_person_ID', 'Delivery_person_Ratings']]
                                          .groupby('Delivery_person_ID')
                                          .mean()
                                          .reset_index())
            st.dataframe(df_avg_ratings_per_deliver)
            
        with col2:
            st.markdown('##### Avaliação média por trânsito')
            df_avg_std_rating_by_traffic = ((df1.loc[:, ['Delivery_person_Ratings', 'Road_traffic_density']]
                                            .groupby('Road_traffic_density')
                                            .agg({'Delivery_person_Ratings':['mean', 'std']})))
            # mudança no nome das colunas
            df_avg_std_rating_by_traffic.columns = ['Delivery_mean', 'Delivery_std']
            # reset do index
            df_avg_std_rating_by_traffic.reset_index()
            # exibir variável 
            st.dataframe(df_avg_std_rating_by_traffic)
            
            
            st.markdown('##### Avaliação média por condições climáticas')
            df_avg_std_weatherconditions = (df1.loc[:, ['Delivery_person_Ratings', 'Weatherconditions']]
                                                .groupby('Weatherconditions')
                                                .agg({'Delivery_person_Ratings':['mean', 'std']}))

            # mudança no nome das colunas
            df_avg_std_weatherconditions.columns = ['Delivery_mean', 'Delivery_std']
            # reset index
            df_avg_std_weatherconditions.reset_index()
            # exibir variável 
            st.dataframe(df_avg_std_weatherconditions)
        
    with st.container():
        st.markdown('''---''')
        st.title('Velocidade de Entrega')
        
        col1, col2 = st.columns(2, gap='large')
        
        with col1:
            st.markdown('##### Top entregadores mais rápidos')
            df2 = top_delivers(df1, top_asc=True) 
            st.dataframe(df2)
                
        with col2:
            st.markdown('##### Top entregadores mais lentos')
            df2 = top_delivers(df1, top_asc=False)
            st.dataframe(df2)   


          