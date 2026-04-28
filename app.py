import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Coffee Demand Forecasting",
    layout="wide"
)

# ---------------------------
# TITLE
# ---------------------------
st.title("Coffee Demand Forecasting Dashboard")
st.markdown("Analyze demand patterns, peak hours, and store performance")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_excel("Afficionado Coffee Roasters.xlsx")

df = load_data()

# ---------------------------
# DATA PREPROCESSING
# ---------------------------
df['transaction_time'] = pd.to_datetime(df['transaction_time'], format='%H:%M:%S')

df['datetime'] = pd.to_datetime(
    df['year'].astype(str) + ' ' + df['transaction_time'].astype(str)
)

df = df.sort_values('datetime')

df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.dayofweek
df['date'] = df['datetime'].dt.date

df['revenue'] = df['transaction_qty'] * df['unit_price']

# ---------------------------
# SIDEBAR FILTER
# ---------------------------
st.sidebar.header("Filters")

stores = df['store_id'].unique()
selected_store = st.sidebar.selectbox("Select Store", stores)

df_store = df[df['store_id'] == selected_store]

# ---------------------------
# KPI SECTION
# ---------------------------
st.subheader("Key Metrics")

total_revenue = df_store['revenue'].sum()
total_transactions = df_store['transaction_qty'].sum()
avg_transactions = df_store['transaction_qty'].mean()

col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"${total_revenue:,.0f}")
col2.metric("Total Transactions", f"{total_transactions:,}")
col3.metric("Average Transactions", f"{avg_transactions:.2f}")

# ---------------------------
# PEAK HOUR ANALYSIS
# ---------------------------
st.subheader("Peak Hour Analysis")

hourly = df_store.groupby('hour')['transaction_qty'].sum()

fig, ax = plt.subplots()
hourly.plot(kind='bar', ax=ax)
ax.set_title("Transactions by Hour")
ax.set_xlabel("Hour")
ax.set_ylabel("Transactions")
st.pyplot(fig)

# ---------------------------
# HEATMAP
# ---------------------------
st.subheader("Demand Heatmap")

heatmap_data = df_store.pivot_table(
    values='transaction_qty',
    index='day',
    columns='hour',
    aggfunc='sum'
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(heatmap_data, cmap='coolwarm', ax=ax)
ax.set_title("Day vs Hour Demand")
st.pyplot(fig)

# ---------------------------
# DAILY REVENUE TREND
# ---------------------------
st.subheader("Daily Revenue Trend")

daily_revenue = df_store.groupby('date')['revenue'].sum()

fig, ax = plt.subplots()
daily_revenue.plot(ax=ax)
ax.set_title("Daily Revenue")
ax.set_xlabel("Date")
ax.set_ylabel("Revenue")
st.pyplot(fig)

# ---------------------------
# STORE COMPARISON
# ---------------------------
st.subheader("Store Comparison")

store_perf = df.groupby('store_id')['revenue'].sum()

fig, ax = plt.subplots()
store_perf.plot(kind='bar', ax=ax)
ax.set_title("Revenue by Store")
st.pyplot(fig)

# ---------------------------
# FORECAST (MOVING AVERAGE)
# ---------------------------
st.subheader("Demand Forecast (7-Day Moving Average)")

daily_qty = df_store.groupby('date')['transaction_qty'].sum()
rolling = daily_qty.rolling(7).mean()

fig, ax = plt.subplots()
daily_qty.plot(label="Actual", ax=ax)
rolling.plot(label="7-Day Average", ax=ax)
ax.legend()
ax.set_title("Forecast vs Actual")
st.pyplot(fig)

# ---------------------------
# SCENARIO ANALYSIS
# ---------------------------
st.subheader("Scenario Analysis")

base = rolling.dropna()
high = base * 1.2
low = base * 0.8

fig, ax = plt.subplots()
base.plot(label="Base", ax=ax)
high.plot(label="High (+20%)", linestyle='--', ax=ax)
low.plot(label="Low (-20%)", linestyle='--', ax=ax)
ax.legend()
ax.set_title("Scenario Forecast")
st.pyplot(fig)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown("Built using Streamlit")