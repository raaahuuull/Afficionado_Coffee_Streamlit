import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

st.set_page_config(layout="wide")

# -------------------------------
# LOAD DATA (GitHub + Upload)
# -------------------------------
st.sidebar.title("Coffee Dashboard")

use_github = st.sidebar.checkbox("Use GitHub Dataset (Default)", value=True)

GITHUB_URL = "https://raw.githubusercontent.com/raaahuuull/Afficionado_Coffee_Streamlit/main/Afficionado%20Coffee%20Roasters.xlsx"

if use_github:
    try:
        df_raw = pd.read_excel(GITHUB_URL)
        st.sidebar.success("Loaded from GitHub")
    except:
        st.sidebar.error("GitHub load failed")
        df_raw = None
else:
    uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])
    if uploaded_file:
        df_raw = pd.read_excel(uploaded_file)
    else:
        df_raw = None

if df_raw is None:
    st.warning("Please upload dataset or enable GitHub option")
    st.stop()

# -------------------------------
# DATA CLEANING
# -------------------------------
df = df_raw.copy()

# Convert time safely
df['transaction_time'] = pd.to_datetime(df['transaction_time'], errors='coerce')

# Extract clean time
df['time_str'] = df['transaction_time'].dt.strftime('%H:%M:%S')

# Combine with year safely
df['datetime'] = pd.to_datetime(
    df['year'].astype(str) + ' ' + df['time_str'],
    errors='coerce'
)

df = df.dropna(subset=['datetime'])

# Features
df['hour'] = df['datetime'].dt.hour
df['date'] = df['datetime'].dt.date
df['day'] = df['datetime'].dt.dayofweek
df['revenue'] = df['transaction_qty'] * df['unit_price']

# -------------------------------
# SIDEBAR FILTER
# -------------------------------
stores = df['store_location'].unique().tolist()
selected_store = st.sidebar.selectbox("Select Store", stores)

df = df[df['store_location'] == selected_store]

# -------------------------------
# MAIN DASHBOARD
# -------------------------------
st.title("Coffee Demand Forecasting Dashboard")

# Metrics
col1, col2, col3, col4 = st.columns(4)

total_revenue = df['revenue'].sum()
total_transactions = len(df)
avg_order = df['revenue'].mean()

# Safe peak hour
if df['hour'].dropna().empty:
    peak_hour = "N/A"
else:
    peak_hour = int(df.groupby('hour').size().idxmax())

col1.metric("Revenue", f"${total_revenue:,.0f}")
col2.metric("Transactions", f"{total_transactions:,}")
col3.metric("Avg Order", f"${avg_order:.2f}")
col4.metric("Peak Hour", f"{peak_hour}:00")

# -------------------------------
# DAILY TREND
# -------------------------------
st.subheader("Daily Revenue Trend")

daily = df.groupby(['date', 'store_location'])['revenue'].sum().reset_index()

st.line_chart(
    daily.pivot(index='date', columns='store_location', values='revenue')
)

# -------------------------------
# PEAK HOUR ANALYSIS
# -------------------------------
st.subheader("Peak Hour Analysis")

hourly = df.groupby('hour').size()

if not hourly.empty:
    fig, ax = plt.subplots()
    hourly.plot(kind='bar', ax=ax)
    ax.axhline(hourly.mean(), color='red', linestyle='--')
    st.pyplot(fig)
else:
    st.warning("No hourly data available")

# -------------------------------
# HEATMAP
# -------------------------------
st.subheader("Demand Heatmap")

heatmap_data = df.pivot_table(
    index='store_location',
    columns='hour',
    values='transaction_id',
    aggfunc='count'
)

if not heatmap_data.empty:
    fig, ax = plt.subplots()
    sns.heatmap(heatmap_data, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
else:
    st.warning("Not enough data for heatmap")

# -------------------------------
# PROPHET FORECAST
# -------------------------------
st.subheader("Demand Forecast")

forecast_days = st.slider("Forecast Days", 7, 60, 30)

daily_store = df.groupby('date')['revenue'].sum().reset_index()
daily_store.columns = ['ds', 'y']
daily_store['ds'] = pd.to_datetime(daily_store['ds'])

# Safety check
if len(daily_store) < 10:
    st.warning("Not enough data for forecasting")
else:
    model = Prophet()
    model.fit(daily_store)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    fig = model.plot(forecast)

    # Reduce red shading
    for coll in fig.gca().collections:
        coll.set_alpha(0.2)

    st.pyplot(fig)

    st.write("Latest Prediction:", round(forecast['yhat'].iloc[-1], 2))
