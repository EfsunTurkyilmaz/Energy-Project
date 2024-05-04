
import streamlit as st
from time_series_TR import main_TR
from il_zaman_serisi import main_il
from Data_Overview import main_data
import pandas as pd
from datetime import timedelta
import folium
from folium.plugins import MarkerCluster
import datetime
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import bar_chart_race as bcr
from streamlit_folium import folium_static
import branca
import warnings
warnings.filterwarnings('ignore')

# Ayarlar
sns.set_style('whitegrid')
st.set_page_config(layout="wide")
# Streamlit uygulamanızın başında bu CSS stilini ekleyin
custom_css = """
<style>
/* Sekme başlıklarının stilini büyüt ve özelleştir */
.css-1siy2j7 {
    font-size: 1.5rem !important;
    font-weight: bold !important;
}

/* Diğer başlık ve metin boyutlarını düzenle */
h1, h2, h3, .stDataFrame, .stMarkdown {
    font-size: 1.1rem !important;
}

/* İlerleme çubuğu stilini özelleştir */
.stProgress > div > div > div > div {
    height: 20px !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def set_sidebar_style():
    st.markdown(
        '''
        <style>
        .stSidebar > div:first-child {
            background-color: #f5f5dc;
        }
        </style>
        ''',
        unsafe_allow_html=True
    )
# Custom container component
def custom_container(content):
    st.markdown(
        f"""
        <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
            {content}
        </div>
        """
        , unsafe_allow_html=True
    )
set_sidebar_style()

# Sidebar konfigürasyonu
st.sidebar.title("Menu")
st.set_option('deprecation.showPyplotGlobalUse', False)
uploaded_file = st.sidebar.file_uploader("Veri Dosyasını Yükle", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file),
        else:
            df = pd.read_csv(uploaded_file)
        st.session_state['data'] = df
        st.sidebar.success('Dosya başarıyla yüklendi.')
    except Exception as e:
        st.sidebar.error(f'Dosya yüklenirken hata oluştu: {e}')

# Sidebar konfigürasyonu
page = st.sidebar.selectbox("Menu", ["Home", "Data Overview", "Modelling", "Results", "Contact"])

# Home Sayfası
if page == "Home":
    st.title("Türkiye'nin Enerji Görünümü")
    st.image('/Users/asmir/Desktop/Enerji/Enerji/PHOTOS/roadmap.png', use_column_width='always')
    st.markdown("""
        ## 
        Bu dashboard, Türkiye'nin enerji tüketimini analiz etmek için hazırlanmıştır.
        Detaylı veri analizleri, tahminler ve anormallik tespitleri yapabilirsiniz.
    """)

# Data Overview Sayfası
elif page == "Data Overview" and 'data' in st.session_state:
    st.title("Data Overview")
    st.dataframe(st.session_state['data'].head())
    main_data()

# Modelling Sayfası
elif page == "Modelling":
    if 'data' in st.session_state:
        st.title("Modelling")
        selected_tab = st.sidebar.radio("Seçenekler", ["Türkiye'nin Enerji Tüketim Tahmini", "İl Enerji Tüketim Tahmini"])

        if selected_tab == "Türkiye'nin Enerji Tüketim Tahmini":
            st.subheader("Türkiye'nin Enerji Tüketim Tahmini")
            main_TR()

        elif selected_tab == "İl Enerji Tüketim Tahmini":
            st.subheader("İl Enerji Tüketim Tahmini")
            st.dataframe(st.session_state['data'].head())
            main_il()
    else:
        st.error("Lütfen önce bir veri dosyası yükleyin.")

# Results Sayfası
elif page == "Results":
    st.title("Results")
    st.image('/Users/asmir/Desktop/Enerji/Enerji/PHOTOS/results.png', use_column_width='always')


# Contact Sayfası
elif page == "Contact":
    st.title("Contact")
    st.markdown("""
        ## İletişim Bilgileri
        Bu bölümde, iletişim bilgileri ve form bulunmaktadır.
        Bizimle iletişime geçmek için aşağıdaki ekip üyelerinden birine tıklayabilirsiniz.
    """)
    st.video('/Users/asmir/Desktop/Enerji/Enerji/VIDEOS/VIDEO-2024-04-10-01-50-33.mp4')

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write("Person 1")
    with col2:
        st.write("Person 2")
    with col3:
        st.write("Person 3")
    with col4:
        st.write("Person 4")
    with col5:
        st.write("Person 5")

    with st.form("contact_form"):
        name = st.text_input("İsim")
        email = st.text_input("E-mail")
        message = st.text_area("Mesajınız")
        submitted = st.form_submit_button("Gönder")
        if submitted:
            st.success("Mesajınız başarıyla gönderildi.")
