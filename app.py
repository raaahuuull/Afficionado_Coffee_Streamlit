import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="Coffee Demand Dashboard", layout="wide")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Coffee Dashboard")

use_github = st.sidebar.checkbox("Use GitHub Dataset", value=True)
uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

GITHUB_URL = "https://raw.githubusercontent.com/raaahuuull/Afficionado_Coffee_Streamlit/main/Afficionado%20Coffee%20Roasters.xlsx"

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_github():
    return pd.read_excel(GITHUB_URL)

df_raw = None

if use_github:
    try:
        df_raw = load_github()
        st.sidebar.success("Loaded from GitHub")
    except:
        st.sidebar.error("GitHub load failed")

if uploaded_file:
    df_raw = pd.read_excel(uploaded_file)
    st.sidebar.success("Uploaded file loaded")

if df_raw is None:
    st.warning("Please upload dataset or enable GitHub option")
    st.stop()

# -----------------------------
# DATA CLEANING
# -----------------------------
df = df_raw.copy()

df['transaction_time'] = pd.to_datetime(df['transaction_time'], errors='coerce')
df['time_str'] = df['transaction_time'].dt.strftime('%H:%M:%S')

df['datetime'] = pd.to_datetime(
    df['year'].astype(str) + ' ' + df['time_str'],
    format='%Y %H:%M:%S',
    errors='coerce'
)

df = df.dropna(subset=['datetime'])

df['hour'] = df['datetime'].dt.hour
df = df.dropna(subset=['hour'])

df['date'] = df['datetime'].dt.date
df['revenue'] = df['transaction_qty'] * df['unit_price']

# -----------------------------
# TITLE
# -----------------------------
st.title("Coffee Demand Forecasting Dashboard")
st.caption("Demand analysis and forecasting using time-series methods")

# -----------------------------
# KPIs
# -----------------------------
total_revenue = df['revenue'].sum()
total_transactions = len(df)
avg_order = df['revenue'].mean()

hourly_counts = df.groupby('hour').size()

if hourly_counts.empty:
    peak_hour = "N/A"
else:
    peak_hour = int(hourly_counts.idxmax())

c1, c2, c3, c4 = st.columns(4)

c1.metric("Revenue", f"${total_revenue:,.0f}")
c2.metric("Transactions", f"{total_transactions:,}")
c3.metric("Avg Order", f"${avg_order:.2f}")
c4.metric("Peak Hour", f"{peak_hour}:00")

# -----------------------------
# DAILY TREND
# -----------------------------
st.subheader("Daily Revenue Trend")

daily = df.groupby(['date', 'store_location'])['revenue'].sum().reset_index()
pivot_daily = daily.pivot(index='date', columns='store_location', values='revenue')

st.line_chart(pivot_daily)

# -----------------------------
# PEAK HOURS
# -----------------------------
st.subheader("Peak Hour Analysis")

hourly = df.groupby('hour').size().reindex(range(24), fill_value=0)

fig, ax = plt.subplots()
ax.bar(hourly.index, hourly.values)
ax.axhline(hourly.mean(), linestyle='--', color='red')
ax.set_xlabel("Hour")
ax.set_ylabel("Transactions")

st.pyplot(fig)

# -----------------------------
# HEATMAP
# -----------------------------
st.subheader("Demand Heatmap")

heatmap = df.groupby(['store_location', 'hour']).size().unstack(fill_value=0)

fig2, ax2 = plt.subplots()
im = ax2.imshow(heatmap, aspect='auto')

ax2.set_yticks(range(len(heatmap.index)))
ax2.set_yticklabels(heatmap.index)

ax2.set_xticks(range(24))
ax2.set_xticklabels(range(24))

plt.colorbar(im)

st.pyplot(fig2)

# -----------------------------
# PROPHET FORECAST
# -----------------------------
st.subheader("Forecasting (Prophet)")

forecast_days = st.slider("Forecast Days", 7, 60, 30)

store_list = df['store_location'].unique()
selected_store = st.selectbox("Select Store", store_list)

df_store = df[df['store_location'] == selected_store]

daily_store = df_store.groupby('date')['revenue'].sum().reset_index()
daily_store.columns = ['ds', 'y']

model = Prophet(daily_seasonality=True, weekly_seasonality=True)
model.fit(daily_store)

future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

fig3, ax3 = plt.subplots()

ax3.plot(forecast['ds'], forecast['yhat'], label='Forecast')

ax3.fill_between(
    forecast['ds'],
    forecast['yhat_lower'],
    forecast['yhat_upper'],
    alpha=0.2
)

ax3.set_xlabel("Date")
ax3.set_ylabel("Revenue")
ax3.legend()

st.pyplot(fig3)

# -----------------------------
# MODEL EVALUATION
# -----------------------------
st.subheader("Model Performance")

actual = daily_store['y'].values[-30:]
pred = forecast['yhat'].values[-30:]

mae = np.mean(np.abs(actual - pred))
rmse = np.sqrt(np.mean((actual - pred)**2))

st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")
