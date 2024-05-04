import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    return df
def prepare_data(df):
    df['TARIH'] = pd.to_datetime(df['TARIH'], errors='coerce')
    df.dropna(subset=['TARIH'], inplace=True)
    df.sort_values(by=['ILLER', 'TARIH'], inplace=True)
    df['YEARMONTH'] = df['TARIH'].dt.to_period('M')
    df_grouped = df.groupby(['ILLER', 'YEARMONTH']).agg({'TUKETIM_GENEL_TOPLAM': 'sum'}).reset_index()
    df_grouped['TARIH'] = df_grouped['YEARMONTH'].dt.to_timestamp()
    df_grouped.set_index('TARIH', inplace=True)
    return df_grouped

# SARIMA modeli kurma ve tahmin
def sarima_model_and_forecast(df, city, order, seasonal_order, steps):
    city_data = df[df['ILLER'] == city]
    model = SARIMAX(city_data['TUKETIM_GENEL_TOPLAM'], order=order, seasonal_order=seasonal_order)
    results = model.fit(disp=False)

    # Gerçek değerler
    actuals = city_data['TUKETIM_GENEL_TOPLAM']

    # Tahminler
    forecast = results.get_forecast(steps=steps)
    forecast_df = forecast.conf_int()
    forecast_df['Forecast'] = forecast.predicted_mean
    forecast_df = forecast_df[['Forecast', 'lower TUKETIM_GENEL_TOPLAM', 'upper TUKETIM_GENEL_TOPLAM']]

    # Tahminleri gerçek verilerle birleştir
    combined_df = pd.concat([actuals, forecast_df], axis=1)
    return combined_df, results

# Main app
def main_il():
    if 'data' in st.session_state:
        df = st.session_state['data']
        df_processed = prepare_data(df)
        st.header("Tahmin Edilecek İl")
        city = st.selectbox('İl Seçin', df_processed['ILLER'].unique(),help="Tahmin yapılacak ili seçin. Bu il üzerinden model eğitimi ve tahmin işlemleri gerçekleştirilecektir.")
        st.header("Parametre Seçimi")
        # Assume model parameters are set interactively via Streamlit widgets
        # Model parametreleri
        p = st.number_input(
            'AR derecesi (p)',
            min_value=0,
            max_value=5,
            value=1,
            help="Otomatik regresyon (AR) derecesi, modelin geçmiş değerlerin bağımlılığını ne kadar geriye götüreceğini belirtir. Örneğin, p=2 ise model, tahminleri yaparken son iki gözlemi dikkate alır."
        )
        d = st.number_input(
            'Fark alma derecesi (d)',
            min_value=0,
            max_value=2,
            value=1,
            help="Fark alma derecesi, veri serisinin kaçıncı dereceden farklarının alınacağını ifade eder. Bu, seriyi durağan hale getirmek için kullanılır. d=1 genellikle seriyi bir kez fark alarak durağanlaştırmaya çalışır."
        )
        q = st.number_input(
            'MA derecesi (q)',
            min_value=0,
            max_value=5,
            value=1,
            help="Hareketli ortalama (MA) derecesi, modelin tahmin hatalarının bağımlılığını ne kadar geriye götüreceğini belirtir. q=2 ise, model tahminleri yaparken son iki tahmin hatasını dikkate alır."
        )
        st.header("Mevsimsel Dereceler")
        P = st.slider(
            'Mevsimsel AR derecesi (P)', 0, 5, 1,
            help="Mevsimsel AR derecesi, modelin mevsimsel geçmiş değerlerin bağımlılığını ne kadar geriye götüreceğini belirtir."
        )
        D = st.slider(
            'Mevsimsel fark alma derecesi (D)', 0, 2, 1,
            help="Mevsimsel fark alma derecesi, serinin mevsimsel durağanlığını sağlamak için gereken fark alma işlemlerinin sayısını belirtir."
        )
        Q = st.slider(
            'Mevsimsel MA derecesi (Q)', 0, 5, 1,
            help="Mevsimsel MA derecesi, modelin mevsimsel tahmin hatalarının bağımlılığını ne kadar geriye götüreceğini ifade eder."
        )
        s = st.slider(
            'Mevsimsel periyot (s)', 1, 12, 12,
            help="Mevsimsel periyot, verilerin tekrarlanma sıklığını belirtir. Örneğin, aylık veriler için genellikle s=12 kullanılır."
        )
        steps = st.number_input(
            'Tahmin Adım Sayısı', 1, 24, 12,
            help="Tahmin adım sayısı, modelin ileriye dönük kaç adım tahmin yapacağını belirtir."
        )

        if st.button('Modeli Eğit ve Tahmin Yap'):
            combined_df, results = sarima_model_and_forecast(df_processed, city, (p, d, q), (P, D, Q, s), steps)

            # Gerçek ve tahmin edilen değerlerin görselleştirilmesi
            fig = px.line(combined_df, x=combined_df.index, y='TUKETIM_GENEL_TOPLAM',
                          labels={'value': 'Tüketim', 'index': 'Tarih'},
                          title=f'{city} için Enerji Tüketim Tahminleri')
            fig.update_layout(height=500, width=900, title_font_size=24,
                              font=dict(family="Arial, sans-serif"))  # Başlık boyutu ve yazı tipi ayarı

            # Tahminlerin görselleştirilmesi
            fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['Forecast'], mode='lines', name='Tahmin',
                                     line=dict(color='red', width=2)))
            fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['lower TUKETIM_GENEL_TOPLAM'], mode='lines',
                                     name='Alt Güven Aralığı', fill='tonexty', line=dict(color='lightblue')))
            fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df['upper TUKETIM_GENEL_TOPLAM'], mode='lines',
                                     name='Üst Güven Aralığı', fill='tonexty', line=dict(color='lightblue')))
            st.plotly_chart(fig,use_container_width=True)
            # Tahmin tablosunun oluşturulması
            st.subheader('Tahminler Tablosu')
            st.write(combined_df.tail(steps))  # Sadece tahminleri göster
            # Model özetinin düzenli bir şekilde gösterilmesi

            # Checkbox ile kullanıcının model özetini gösterip göstermemesini kontrol et
            show_summary = st.checkbox("Model Özetini Göster")

            # Model özetini gösterme durumuna göre karar verme
            if show_summary:
                st.subheader('Model Özeti')
                st.text(str(results.summary()))
            # Veri ayrıştırma grafiği
            result = seasonal_decompose(df_processed[df_processed['ILLER'] == city]['TUKETIM_GENEL_TOPLAM'],
                                        model='additive', period=12)
            fig_decomposition = make_subplots(rows=3, cols=1, subplot_titles=("Trend", "Seasonal", "Residual"))
            fig_decomposition.add_trace(
                go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Trend', line=dict(color='red')),
                row=1, col=1)
            fig_decomposition.add_trace(
                go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Seasonal',
                           line=dict(color='green')), row=2, col=1)
            fig_decomposition.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Residual',
                                                   line=dict(color='blue')), row=3, col=1)
            fig_decomposition.update_layout(title=f"{city} için Veri Ayrıştırma Grafiği", height=1000, width=900,
                                            title_font_size=24,
                                            font=dict(family="Arial, sans-serif"))  # Başlık boyutu ve yazı tipi ayarı
            st.plotly_chart(fig_decomposition,use_container_width=True)

    else:
        st.error("Data is not loaded. Please upload data through the main interface.")













