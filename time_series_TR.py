
import warnings
import streamlit as st
import numpy as np
from datetime import timedelta

import time
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, roc_curve, auc
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
import plotly.express as px
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import bar_chart_race as bcr
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import branca
import branca.colormap as cm
warnings.filterwarnings('ignore')

# Settings
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
def load_data(filepath):
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
        return df
    except Exception as e:
        raise e
def create_date_features(df):
    df['MONTH'] = df['TARIH'].dt.month
    df['DAY_OF_MONTH'] = df['TARIH'].dt.day
    df['DAY_OF_YEAR'] = df['TARIH'].dt.dayofyear
    df['WEEK_OF_YEAR'] = df['TARIH'].dt.isocalendar().week
    df['DAY_OF_WEEK'] = df['TARIH'].dt.dayofweek
    df['YEAR'] = df['TARIH'].dt.year
    return df
def random_noise(dataframe):
    return np.random.normal(scale=0.01, size=(len(dataframe),))
def add_lag_and_noise_features(df, lags, column):
    for lag in lags:
        df[f'{column}_LAG_{lag}'] = df.groupby(['ILLER'])[column].transform(
            lambda x: x.shift(lag)) + random_noise(df)
    return df
def add_rolling_mean_features(df, window_sizes, column='TUKETIM_GENEL_TOPLAM'):
    for window in window_sizes:
        for lag in range(1, 13):
            df[f'{column}_ROLLING_MEAN_WINDOW_{window}_LAG_{lag}'] = df.groupby(['ILLER'])[column].shift(lag).rolling(window=window, min_periods=1, win_type="triang").mean()+random_noise(df)
    return df
def add_ewm_features(df, alphas, lags, column='TUKETIM_GENEL_TOPLAM'):
    for alpha in alphas:
        for lag in lags:
            df[f'{column}_EWM_ALPHA_{str(alpha).replace(".", "")}_LAG_{lag}'] = df.groupby(['ILLER'])[column].shift(
                    lag).ewm(alpha=alpha).mean()
    return df
def add_seasonality_features(df):
    df['SIN_MONTH'] = np.sin(2 * np.pi * df['MONTH'] / 12)
    df['COS_MONTH'] = np.cos(2 * np.pi * df['MONTH'] / 12)
    return df
def preprocess_data(df, lags, window_sizes, alphas):
    df = create_date_features(df)
    df = add_lag_and_noise_features(df, lags, 'TUKETIM_GENEL_TOPLAM')
    df = add_rolling_mean_features(df, window_sizes)
    df = add_seasonality_features(df)
    df = add_ewm_features(df, alphas, lags, 'TUKETIM_GENEL_TOPLAM')
    return df
def split_data(df, split_date):
    df_train = df[df['TARIH'] < split_date]
    df_val = df[(df['TARIH'] >= split_date) & (df['TARIH'] < pd.Timestamp(split_date).replace(month=pd.Timestamp(split_date).month+3))]
    df_test = df[df['TARIH'] >= pd.Timestamp(split_date).replace(month=pd.Timestamp(split_date).month+3)]
    return df_train, df_val, df_test
# Dinamik parametre seçimi
def get_dynamic_params(n_samples):
    if n_samples < 1000:
        params = {'n_estimators': 100, 'learning_rate': 0.1}
    elif n_samples < 10000:
        params = {'n_estimators': 300, 'learning_rate': 0.05}
    else:
        params = {'n_estimators': 500, 'learning_rate': 0.01}
    return params
def train_model(df_train, df_val):
    # Özellik ve hedef değişkenlerini ayır
    features = [col for col in df_train.columns if col not in ['TARIH', 'ILLER', 'TUKETIM_GENEL_TOPLAM',"TUKETIM_PAYI(%)","BOLGE"]]
    X_train = df_train[features]
    y_train = df_train['TUKETIM_GENEL_TOPLAM']
    X_val = df_val[features]
    y_val = df_val['TUKETIM_GENEL_TOPLAM']
    # Sabit parametreleri tanımla
    other_params = {
        'max_depth': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    # Modeli eğitirken parametreleri al
    dynamic_params = get_dynamic_params(len(X_train))

    # Modeli oluştur ve eğit
    model = lgb.LGBMRegressor(**dynamic_params, **other_params)
    early_stopping = lgb.callback.early_stopping(stopping_rounds=100, verbose=0)
    # Tahmin yap ve perform
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='smape', callbacks=[early_stopping])
    # Model değerlendirme
    y_pred_val = model.predict(X_val)
    smape = np.mean(2 * np.abs(y_val - y_pred_val) / (np.abs(y_val) + np.abs(y_pred_val))) * 100
    mae = mean_absolute_error(y_val, y_pred_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    # Model ve metrikleri session_state'e kaydet
    st.session_state['model'] = model
    st.session_state['features'] = features
    st.session_state['smape'] = smape
    st.session_state['mae'] = mae
    st.session_state['rmse'] = rmse
    return model, smape, mae, rmse
def display_metrics():
    # Metrikleri göster
    if 'smape' in st.session_state and 'mae' in st.session_state and 'rmse' in st.session_state:
        st.write(f"SMAPE: {st.session_state['smape']:.2f}%")
        st.write(f"MAE: {st.session_state['mae']:.2f}")
        st.write(f"RMSE: {st.session_state['rmse']:.2f}")

def validate_parameters(months_ahead):
    if not 1 <= months_ahead <= 12:
        raise ValueError("Months for prediction must be between 1 and 12.")

    # Alphas should always be between 0 and 1. We limit the maximum value to 0.9 to avoid reaching 1.
    alphas = [min(0.1 * i, 0.9) for i in range(1, months_ahead + 1)]

    # Lags and window_sizes should be between 1 and the number of months_ahead.
    lags = list(range(1, months_ahead + 1))
    window_sizes = list(range(1, months_ahead + 1))

    return alphas, lags, window_sizes

# Use the function and handle any ValueError exceptions
try:
    alphas, lags, window_sizes = validate_parameters(13)  # An intentionally incorrect value for demonstration
except ValueError as error:
    print(error)  # Output the error message
# This is how you would normally call the function with a correct value
try:
    alphas, lags, window_sizes = validate_parameters(12)  # This should be correct
    print("Validated parameters:", alphas, lags, window_sizes)
except ValueError as error:
    print(error)

def forecast(model, df_test):
    features = [col for col in df_test.columns if col not in ['TARIH', 'ILLER', 'TUKETIM_GENEL_TOPLAM',"TUKETIM_PAYI(%)","BOLGE"]]
    X_test = df_test[features]
    predictions = model.predict(X_test)
    df_test['PREDICTIONS'] = predictions
    return df_test

def generate_future_dates(last_date, months_ahead):
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=months_ahead, freq='M')
    future_df = pd.DataFrame(future_dates, columns=['TARIH'])
    future_df = create_date_features(future_df)
    future_df = add_seasonality_features(future_df)

    # Simulate other features as needed, possibly using assumed values or statistical summaries from historical data
    # Add dummy implementations for lag and rolling features if they don't depend on sequential data
    # Example:
    for col in ['some_lag_feature', 'some_rolling_feature']:
        future_df[col] = 0  # Or some other placeholder or statistically derived value

    return future_df

def make_predictions(model, future_df):
    # Tahmin için gerekli özellikleri seç
    features = [col for col in future_df.columns if col not in ['TARIH', 'TUKETIM_GENEL_TOPLAM', 'ILLER', "TUKETIM_PAYI(%)", "BOLGE"]]
    predictions = model.predict(future_df[features])
    future_df['PREDICTIONS'] = predictions
    return future_df


def plot_future_predictions(predicted_df):
    plt.figure(figsize=(15, 5))
    plt.plot(predicted_df['TARIH'], predicted_df['PREDICTIONS'], label='Forecasted Energy Consumption')
    plt.title('Future Energy Consumption Forecast')
    plt.xlabel('Date')
    plt.ylabel('Energy Consumption')
    plt.legend()
    plt.show()
# Mevcut veri setinizin son tarihi



def display_predictions_table(df):
    return df[['TARIH', 'TUKETIM_GENEL_TOPLAM', 'PREDICTIONS']]
def plot_forecast(df_test):
    plt.figure(figsize=(15, 5))
    plt.plot(df_test['TARIH'], df_test['TUKETIM_GENEL_TOPLAM'], label='Actual')
    plt.plot(df_test['TARIH'], df_test['PREDICTIONS'], label='Forecast')
    plt.title('Energy Consumption Forecast')
    plt.xlabel('Date')
    plt.ylabel('Energy Consumption')
    plt.legend()
    plt.show()
def plot_decomposition(df, column='TUKETIM_GENEL_TOPLAM', model='additive'):
    # df['TARIH'] sütununu tarih indeksi olarak ayarla
    df = df.set_index('TARIH')
    decomposition = seasonal_decompose(df[column], model=model, period=12)

    fig_decomposition, axs = plt.subplots(4, 1, figsize=(14, 10))  # Grafik boyutunu büyüt

    colors = ['skyblue', 'sandybrown', 'limegreen', 'violet']  # Renkler

    axs[0].plot(decomposition.trend.index, decomposition.trend, label='Trend', color=colors[0])
    axs[0].legend(loc='upper left')
    axs[0].set_title('Trend Component')

    axs[1].plot(decomposition.seasonal.index, decomposition.seasonal, label='Seasonality', color=colors[1])
    axs[1].legend(loc='upper left')
    axs[1].set_title('Seasonal Component')

    axs[2].plot(decomposition.resid.index, decomposition.resid, label='Residuals', color=colors[2])
    axs[2].legend(loc='upper left')
    axs[2].set_title('Residual Component')

    axs[3].plot(df.index, df[column], label='Observed', color=colors[3])
    axs[3].legend(loc='upper left')
    axs[3].set_title('Observed Data')
    axs[3].set_xlim(df.index.min(), df.index.max())  # Eksenleri sınırlarını ayarla

    plt.tight_layout()
def plot_feature_importance(model, features, num_features=10):
    # Modelden özellik önem derecelerini al
    importances = model.feature_importances_

    # Özellik önem derecelerini DataFrame olarak kaydet
    fi_df = pd.DataFrame({'Features': features, 'Importance': importances})
    fi_df.sort_values(by='Importance', ascending=False, inplace=True)

    # En önemli özellikleri seç
    fi_df_top = fi_df.head(num_features)

    # Grafik çiz
    plt.figure(figsize=(10, 5))
    sns.barplot(data=fi_df_top, x='Importance', y='Features', palette='viridis')
    plt.title('Top Feature Importances')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()

    # Streamlit'te grafik ve tablo göster
    st.pyplot(plt)
    st.write("Top Feature Importances:")
    st.dataframe(fi_df_top)
def simulate_training():
    st.write("Training model...")
    progress_bar_train = st.progress(0)
    for percent_complete in range(0, 101, 10):
        time.sleep(0.1)
        progress_bar_train.progress(percent_complete)
    st.success("Model training completed successfully.")
# Function to simulate prediction
def simulate_prediction():
    st.write("Making predictions...")
    progress_bar_prediction = st.progress(0)
    for percent_complete in range(0, 101, 10):
        time.sleep(0.1)
        progress_bar_prediction.progress(percent_complete)
    st.success("Prediction completed successfully.")
def detect_anomalies(df):
    df_anomalies = df.copy()
    isolation_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    # Tüm veriye uygulanıyor ve -1 veya 1 değerlerini döndürüyor
    df_anomalies['ANOMALIES'] = isolation_forest.fit_predict(df_anomalies[['TUKETIM_GENEL_TOPLAM']])
    df_anomalies['SCORES'] = isolation_forest.decision_function(df_anomalies[['TUKETIM_GENEL_TOPLAM']])
    df_anomalies['DETECTED_CITIES'] = df_anomalies.apply(
        lambda x: x['ILLER'] if x['ANOMALIES'] == -1 else np.nan, axis=1
    )
    # Burada '-1' ise 'Anomaly', '1' ise 'Normal' olarak işaretliyoruz
    df_anomalies['Anomaly Type'] = df_anomalies['ANOMALIES'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
    return df_anomalies

# Bu fonksiyonu çağırmadan önce `detect_anomalies` fonksiyonunu çağrıp 'Anomaly Type' sütununu oluşturduğunuzdan emin olun.

def plot_anomalies(anomalies_df):
    fig = px.scatter(
        anomalies_df,
        x='TARIH',
        y='TUKETIM_GENEL_TOPLAM',
        color='Anomaly Type',  # Renk ayarlama burada yapılıyor
        color_discrete_map={'Anomaly': 'orange', 'Normal': 'pink'}  # Anomali ve Normal durumlar için renkleri belirle
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color="white"
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')))
    return fig

def anomalies_table(df_anomalies):
    df_anomalies = detect_anomalies(df_anomalies)
    # Sadece anomalileri içeren girişleri filtrele
    df_anomalies = df_anomalies[df_anomalies['ANOMALIES'] == -1]
    # Eğer hiç anormali tespit edilmediyse boş DataFrame döndür
    if df_anomalies.empty:
        return df_anomalies
    df_anomalies_filtered = df_anomalies[['TARIH', 'TUKETIM_GENEL_TOPLAM', 'ANOMALIES', 'DETECTED_CITIES', 'ENLEM', 'BOYLAM', 'SCORES']]
    return df_anomalies_filtered


def create_anomaly_map(df_anomalies_filtered, lat_col='ENLEM', lon_col='BOYLAM', city_col='DETECTED_CITIES', anomaly_score_col='ANOMALIES'):
    center_lat = df_anomalies_filtered[lat_col].mean()
    center_lon = df_anomalies_filtered[lon_col].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles='CartoDB positron')
    marker_cluster = MarkerCluster().add_to(m)
    color_scale = cm.linear.YlOrRd_09.scale(df_anomalies_filtered[anomaly_score_col].min(),
                                            df_anomalies_filtered[anomaly_score_col].max())
    color_scale.caption = 'Anomaly Intensity Scale'
    m.add_child(color_scale)

    for _, row in df_anomalies_filtered.iterrows():
        if not pd.isna(row[city_col]):  # Sadece anomali olarak işaretlenmiş şehirler için
            popup_content = folium.Html(f'''
                <b>City:</b> {row[city_col]}<br>
                <b>Score:</b> {row['SCORES']:.2f}<br>
                <b>Latitude:</b> {row[lat_col]}<br>
                <b>Longitude:</b> {row[lon_col]}''', script=True)
            popup = folium.Popup(popup_content, max_width=2650)
            folium.Marker(
                [row[lat_col], row[lon_col]],
                popup=popup,
                icon=folium.Icon(color=color_scale(row['SCORES']))  # Burada skorlara göre renk veriliyor
            ).add_to(marker_cluster)

    return m
def set_custom_css():
    custom_css = """
        <style>
            /* Sekme başlıklarını özelleştir */
            .stTabs > .tab > button {
                font-size: 2rem !important;  /* Boyutu artır */
                padding: 1rem !important;
            }

            /* Sekme içi büyük başlıkların rengini özelleştir */
            .stTabContent > div > div > h1 {
                color: #ff4b4b !important;  /* Kırmızı tonu */
            }

            .stTabContent > div > div > h2 {
                color: #007bff !important;  /* Mavi tonu */
            }

            /* Bilgilendirme mesajı kutusunun rengini açık yeşil yap */
            .stAlert-success {
                background-color: #b8f0b8 !important;  /* Açık yeşil tonu */
            }

            /* İlerleme çubuğu boyutunu ve rengini düzenle */
            .stProgress > div > div > div > div {
                height: 20px !important;
                background-color: #76b5c5 !important;  /* Mavi-yeşil tonu */
            }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def main_TR():
    set_custom_css()

    if 'data' in st.session_state:
        st.dataframe(st.session_state['data'].head())
        df = st.session_state['data']
        df.columns = [col.upper() for col in df.columns]
        df['TARIH'] = pd.to_datetime(df['TARIH'])
        df['ILLER'] = df['ILLER'].astype('category')

        st.write("Preprocessing data...")
        progress_bar_preprocess = st.progress(0)
        for percent_complete in range(0, 101, 10):
            time.sleep(0.1)
            progress_bar_preprocess.progress(percent_complete)
        st.success("Data preprocessing completed.")

        selected_month = st.slider("Select number of months for prediction:", 1, 12, 3)
        alphas, lags, window_sizes = validate_parameters(selected_month)
        df_preprocessed = preprocess_data(df, lags, window_sizes, alphas)

        df_train, df_val, df_test = split_data(df_preprocessed, '2023-01-01')

        model, smape, mae, rmse = train_model(df_train, df_val)
        df_forecasted = forecast(model, df_test)

        tab2, tab1 = st.tabs(["Anomaly Detection", "Time Series and Predictions"])

        with tab2:
            # Anomaly detection bölümüne ilerleme çubuğu ve açıklama ekleme
            st.write("Anomaly detection process is starting...")
            progress_bar_anomaly = st.progress(0)

            # İşlemin ne olduğunu açıklayan bir expander ekleyin
            with st.expander("What is happening now?"):
                st.write("""
                            The anomaly detection process involves several steps:
                            - We are currently running an Isolation Forest algorithm over the preprocessed data.
                            - This algorithm will attempt to identify any data points that deviate significantly from the rest.
                            - Anomalies are identified based on the score from the algorithm.
                            - Once detected, anomalies are plotted for visual inspection.
                        """)

            # Anomaly detection yapılırken ilerlemeyi gösterme
            for percent_complete in range(0, 101, 20):
                time.sleep(0.5)  # Bu da bir simülasyondur ve gerçek veri işlem süresine göre ayarlanmalı
                progress_bar_anomaly.progress(percent_complete)

            st.write("## Anomaly Detection")
            anomalies_df = detect_anomalies(df_preprocessed)
            # İlerleme çubuğunu tamamlama
            progress_bar_anomaly.progress(100)
            st.success("Anomaly detection completed.")

            df_anomalies_filtered = anomalies_table(anomalies_df)
            if not anomalies_df.empty:
                st.session_state['df_anomalies_filtered'] = anomalies_df
                if st.button('Show Anomalies Visualization'):
                    fig_anomalies = plot_anomalies(anomalies_df)
                    st.plotly_chart(fig_anomalies, use_container_width=True)
                if st.button('Show Anomaly Table'):
                    st.subheader('Anomaly Scores with Coordinates')
                    st.dataframe(anomalies_df[['TARIH', 'TUKETIM_GENEL_TOPLAM', 'ANOMALIES', 'DETECTED_CITIES', 'ENLEM', 'BOYLAM']])
                if st.radio("Show map:", ['Yes', 'No'], index=1) == 'Yes':
                    anomaly_map = create_anomaly_map(st.session_state['df_anomalies_filtered'])
                    folium_static(anomaly_map, width=1350)
            else:
                st.error("No anomalies detected or filtered.")

        with tab1:
            st.write("## Time Series Analysis and Predictions")
            display_metrics()
            st.pyplot(plot_forecast(df_forecasted))

            if st.checkbox('Show Prediction Table'):
                st.dataframe(display_predictions_table(df_forecasted))

            if st.checkbox('Show Decomposition'):
                st.pyplot(plot_decomposition(df_preprocessed))

            # Veri bölme
            model = st.session_state.get('model')

            """
            future_df = st.session_state.get(df_preprocessed)  # Verileri yükleyin veya uygun şekilde ayarlayın
            last_date = df['TARIH'].max()
            months_ahead = 12  # Gelecek 12 ay için tahmin yapılacak
            future_df = generate_future_dates(last_date, months_ahead)
            predicted_df = make_predictions(model, future_df)
            plot_future_predictions(predicted_df)
            # Streamlit kod örneği
            st.write("Future Energy Consumption Predictions")
            st.line_chart(predicted_df.set_index('TARIH')['PREDICTIONS'])

            
            """
            st.title("Model Performance and Feature Importance")
            if 'model' in st.session_state:
                num_features = st.number_input('Select number of top features to display', min_value=1, max_value=len(st.session_state['features']), value=10)
                plot_feature_importance(st.session_state['model'], st.session_state['features'], num_features)

    else:
        st.error("Data is not loaded. Please upload data through the main interface.")

def compare_features(df_train, df_test):
    train_features = set(df_train.columns)
    test_features = set(df_test.columns)
    common_features = train_features.intersection(test_features)
    missing_in_test = train_features - test_features
    extra_in_test = test_features - train_features
    print("Common features in both sets:", common_features)
    print("Missing features in test set:", missing_in_test)
    print("Extra features in test set:", extra_in_test)

def check_feature_consistency(df_train, df_val, df_test):
    train_features = set(df_train.columns)
    val_features = set(df_val.columns)
    test_features = set(df_test.columns)

    if not (train_features == val_features == test_features):
        print("Inconsistency in features:")
        print("Features missing in validation set:", train_features - val_features)
        print("Features missing in test set:", train_features - test_features)
        print("Unexpected features in validation set:", val_features - train_features)
        print("Unexpected features in test set:", test_features - train_features)
    else:
        print("All features are consistent across datasets.")
