import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb

# Optional Prophet
try:
    from prophet import Prophet
except:
    Prophet = None

st.set_page_config(page_title="Coffee Demand Forecasting", layout="wide")

# ---------------------------
# CONFIG
# ---------------------------
GITHUB_DATA_URL = "https://raw.githubusercontent.com/raaahuuull/Afficionado_Coffee_Streamlit/main/Afficionado%20Coffee%20Roasters.xlsx"

# ---------------------------
# LOAD DATA (HYBRID)
# ---------------------------
st.sidebar.title("Coffee Dashboard")

use_github = st.sidebar.checkbox("Use GitHub Dataset (Default)", value=True)
uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

@st.cache_data
def load_data(source):
    df = pd.read_excel(source)

    # Extract hour safely
    df['hour'] = df['transaction_time'].apply(
        lambda x: x.hour if hasattr(x, 'hour') else int(str(x).split(':')[0])
    )

    # Create realistic timeline (important fix)
    df = df.sort_values('transaction_id').reset_index(drop=True)
    df['day_flag'] = (df['hour'] < df['hour'].shift(1)).astype(int)
    df['day_index'] = df['day_flag'].cumsum()

    df['date'] = pd.Timestamp('2025-01-01') + pd.to_timedelta(df['day_index'], unit='D')

    # Revenue
    df['revenue'] = df['transaction_qty'] * df['unit_price']

    return df

# Decide data source
if use_github:
    try:
        df = load_data(GITHUB_DATA_URL)
        st.sidebar.success("Loaded from GitHub")
    except:
        st.sidebar.error("GitHub load failed. Upload file.")
        st.stop()
else:
    if uploaded_file is None:
        st.warning("Upload dataset to continue")
        st.stop()
    df = load_data(uploaded_file)

# ---------------------------
# KPIs
# ---------------------------
st.title("Coffee Demand Forecasting Dashboard")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Revenue", f"${df['revenue'].sum():,.0f}")
c2.metric("Transactions", f"{len(df):,}")
c3.metric("Avg Order", f"${df['revenue'].mean():.2f}")
peak_hour = df.groupby('hour')['transaction_qty'].sum().idxmax()
c4.metric("Peak Hour", f"{peak_hour}:00")

st.markdown("---")

# ---------------------------
# DAILY TREND
# ---------------------------
st.subheader("Daily Revenue Trend")

daily = df.groupby(['date','store_location'])['revenue'].sum().reset_index()

if not daily.empty:
    fig = px.line(daily, x='date', y='revenue', color='store_location')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No daily data")

# ---------------------------
# PEAK HOUR ANALYSIS
# ---------------------------
st.subheader("Peak Hour Analysis")

hourly = df.groupby('hour')['transaction_qty'].sum()

fig, ax = plt.subplots()

if hourly.empty:
    st.warning("No hourly data")
else:
    hourly.plot(kind='bar', ax=ax, color='skyblue')
    ax.axhline(hourly.mean(), color='red', linestyle='--')
    ax.set_title("Transactions by Hour")
    st.pyplot(fig)

# ---------------------------
# HEATMAP
# ---------------------------
st.subheader("Demand Heatmap")

heatmap = df.pivot_table(
    values='transaction_qty',
    index='store_location',
    columns='hour',
    aggfunc='sum'
)

if not heatmap.empty:
    fig, ax = plt.subplots(figsize=(10,4))
    sns.heatmap(heatmap, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# ---------------------------
# PEAK DAYS
# ---------------------------
st.subheader("Peak Demand Days")

daily_total = df.groupby('date')['revenue'].sum()
threshold = daily_total.quantile(0.9)
peak_days = daily_total[daily_total > threshold]

st.write(f"Top {len(peak_days)} Peak Days")

fig, ax = plt.subplots()
daily_total.plot(ax=ax)
ax.axhline(threshold, color='red', linestyle='--')
st.pyplot(fig)

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
daily_ts = df.groupby(['date','store_location'])['revenue'].sum().reset_index()

def add_features(df):
    df = df.sort_values(['store_location','date'])
    g = df.groupby('store_location')['revenue']

    df['lag_1'] = g.shift(1)
    df['lag_7'] = g.shift(7)
    df['rolling_7'] = g.transform(lambda x: x.shift(1).rolling(7).mean())
    df['rolling_14'] = g.transform(lambda x: x.shift(1).rolling(14).mean())

    df['day'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day'].isin([5,6]).astype(int)

    le = LabelEncoder()
    df['store_enc'] = le.fit_transform(df['store_location'])

    return df.dropna()

daily_fe = add_features(daily_ts)

# ---------------------------
# MODEL TRAINING
# ---------------------------
st.subheader("Model Performance")

if st.button("Run Models"):

    train = daily_fe.iloc[:-30]
    test = daily_fe.iloc[-30:]

    X_train = train[['lag_1','lag_7','rolling_7','rolling_14','day','is_weekend','store_enc']]
    y_train = train['revenue']

    X_test = test[['lag_1','lag_7','rolling_7','rolling_14','day','is_weekend','store_enc']]
    y_test = test['revenue']

    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    st.write(f"MAE: {mae:.2f}")
    st.write(f"RMSE: {rmse:.2f}")

    fig, ax = plt.subplots()
    ax.plot(y_test.values, label="Actual")
    ax.plot(preds, label="Predicted")
    ax.legend()
    st.pyplot(fig)

# ---------------------------
# PROPHET FORECAST
# ---------------------------
if Prophet is not None:
    st.subheader("Prophet Forecast")

    store = st.selectbox("Select Store", df['store_location'].unique())
    temp = daily_ts[daily_ts['store_location'] == store]

    if not temp.empty:
        p_df = temp.rename(columns={'date':'ds','revenue':'y'})

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            seasonality_mode='multiplicative'
        )

        model.fit(p_df)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        fig = px.line(forecast, x='ds', y='yhat', title="Forecast")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Prophet not available")

st.markdown("---")
st.caption("Data-driven demand forecasting dashboard")
