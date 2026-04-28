import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Safe Prophet import
try:
    from prophet import Prophet
except:
    Prophet = None

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

# CONFIG
st.set_page_config(page_title="Coffee Forecasting", layout="wide")

PALETTE = ['#2563EB', '#16A34A', '#DC2626']
STORES = ['Lower Manhattan', "Hell's Kitchen", 'Astoria']

# SIDEBAR
st.sidebar.title("Coffee Dashboard")
uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

if uploaded_file is None:
    st.warning("Upload dataset to continue")
    st.stop()

# LOAD DATA
@st.cache_data
def load_data(file):
    df = pd.read_excel(file)

    # Extract hour safely
    df['hour'] = df['transaction_time'].apply(
        lambda x: x.hour if hasattr(x, 'hour') else int(str(x).split(':')[0])
    )

    # Create fake continuous date (fix your dataset issue)
    df = df.sort_values('transaction_id').reset_index(drop=True)
    df['day_flag'] = (df['hour'] < df['hour'].shift(1)).astype(int)
    df['day_index'] = df['day_flag'].cumsum()

    df['date'] = pd.Timestamp('2025-01-01') + pd.to_timedelta(df['day_index'], unit='D')
    df['revenue'] = df['transaction_qty'] * df['unit_price']

    return df

df = load_data(uploaded_file)

# =========================
# KPI SECTION
# =========================
st.title("Coffee Demand Forecasting Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Revenue", f"${df['revenue'].sum():,.0f}")
col2.metric("Transactions", f"{len(df):,}")
col3.metric("Avg Order", f"${df['revenue'].mean():.2f}")

st.markdown("---")

# =========================
# DAILY TREND
# =========================
st.subheader("Daily Revenue Trend")

daily = df.groupby(['date', 'store_location'])['revenue'].sum().reset_index()

if not daily.empty:
    fig = px.line(daily, x='date', y='revenue', color='store_location')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No daily data available")

# =========================
# PEAK HOURS
# =========================
st.subheader("Peak Hour Analysis")

hourly = df.groupby('hour')['transaction_id'].count()

fig, ax = plt.subplots()

if hourly.empty:
    st.warning("No hourly data available")
else:
    hourly.plot(kind='bar', ax=ax, color='skyblue')
    ax.axhline(hourly.mean(), color='red', linestyle='--')
    ax.set_title("Transactions by Hour")
    st.pyplot(fig)

# =========================
# FEATURE ENGINEERING
# =========================
daily_ts = df.groupby(['date', 'store_location'])['revenue'].sum().reset_index()

def add_features(df):
    df = df.sort_values(['store_location', 'date'])
    g = df.groupby('store_location')['revenue']

    df['lag_1'] = g.shift(1)
    df['lag_7'] = g.shift(7)
    df['rolling'] = g.transform(lambda x: x.shift(1).rolling(7).mean())

    df['day'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day'].isin([5,6]).astype(int)

    le = LabelEncoder()
    df['store_enc'] = le.fit_transform(df['store_location'])

    return df.dropna()

daily_fe = add_features(daily_ts)

# =========================
# MODEL TRAINING BUTTON
# =========================
st.subheader("Model Training")

if st.button("Run Models"):

    train = daily_fe.iloc[:-30]
    test = daily_fe.iloc[-30:]

    X_train = train[['lag_1','lag_7','rolling','day','is_weekend','store_enc']]
    y_train = train['revenue']
    X_test = test[['lag_1','lag_7','rolling','day','is_weekend','store_enc']]
    y_test = test['revenue']

    # LightGBM
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    st.write("MAE:", round(mae,2))
    st.write("RMSE:", round(rmse,2))

    # Plot predictions
    fig2, ax2 = plt.subplots()
    ax2.plot(y_test.values, label="Actual")
    ax2.plot(preds, label="Predicted")
    ax2.legend()
    st.pyplot(fig2)

# =========================
# PROPHET (OPTIONAL)
# =========================
if Prophet is not None:
    st.subheader("Prophet Forecast")

    store = st.selectbox("Select Store", STORES)
    temp = daily_ts[daily_ts['store_location']==store]

    if not temp.empty:
        p_df = temp.rename(columns={'date':'ds','revenue':'y'})
        m = Prophet()
        m.fit(p_df)

        future = m.make_future_dataframe(periods=30)
        forecast = m.predict(future)

        fig3 = px.line(forecast, x='ds', y='yhat', title="Forecast")
        st.plotly_chart(fig3, use_container_width=True)
else:
    st.warning("Prophet not available in this deployment")
