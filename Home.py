import streamlit as st
from PIL import Image

st.set_page_config(
    page_title='Home',
    page_icon='🕹️'
)


#image_path = '/Users/joaonovaes1/Documents/repos/Fast Track Course - FTC/codigo_jupyter/'
image = Image.open('logo.png')
st.sidebar.image(image, width=120)

st.sidebar.markdown('## Fastest Delivery in Town')
st.sidebar.markdown("""---""")

st.write('# Curry Company Growth Dashboard')
st.markdown("""
        # Como utilizar o DashBoard:
        ### 1. Diferentes Visões
        ### 2. Aulas
        ### 3. Gerenciamento
        
        #### Ask For Help
             @Joaomnovaes1 (discord)
        ## Isso é um teste

""")