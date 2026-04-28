import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="Coffee Dashboard", layout="wide")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Coffee Dashboard")

use_github = st.sidebar.checkbox("Use GitHub Dataset (Default)", value=True)
uploaded_file = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

GITHUB_URL = "https://raw.githubusercontent.com/raaahuuull/Afficionado_Coffee_Streamlit/main/Afficionado%20Coffee%20Roasters.xlsx"

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_github_data():
    return pd.read_excel(GITHUB_URL)

df_raw = None

if use_github:
    try:
        df_raw = load_github_data()
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
df['date'] = df['datetime'].dt.date
df['revenue'] = df['transaction_qty'] * df['unit_price']

# -----------------------------
# TITLE
# -----------------------------
st.title("Coffee Demand Forecasting Dashboard")
st.caption("Analytics + Forecasting + Machine Learning")

# -----------------------------
# METRICS
# -----------------------------
total_revenue = df['revenue'].sum()
total_transactions = len(df)
avg_order = df['revenue'].mean()
peak_hour = df.groupby('hour').size().idxmax()

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
# PEAK HOUR ANALYSIS
# -----------------------------
st.subheader("Peak Hour Analysis")

hourly = df.groupby('hour').size()
hourly = hourly.reindex(range(24), fill_value=0)

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
st.subheader("Prophet Forecast")

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
ax3.legend()

st.pyplot(fig3)

# -----------------------------
# MODEL PERFORMANCE (PROPHET)
# -----------------------------
st.subheader("Prophet Performance")

actual = daily_store['y'].values[-30:]
pred = forecast['yhat'].values[-30:]

mae = np.mean(np.abs(actual - pred))
rmse = np.sqrt(np.mean((actual - pred)**2))

st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")

# -----------------------------
# LIGHTGBM MODEL
# -----------------------------
st.subheader("Machine Learning Model (LightGBM)")

try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import lightgbm as lgb

    df_ml = df.copy()

    le1 = LabelEncoder()
    le2 = LabelEncoder()

    df_ml['store_enc'] = le1.fit_transform(df_ml['store_location'])
    df_ml['product_enc'] = le2.fit_transform(df_ml['product_category'])

    X = df_ml[['hour', 'store_enc', 'product_enc', 'unit_price']]
    y = df_ml['revenue']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05)
    model_lgb.fit(X_train, y_train)

    y_pred = model_lgb.predict(X_test)

    mae_ml = mean_absolute_error(y_test, y_pred)
    rmse_ml = np.sqrt(mean_squared_error(y_test, y_pred))

    c1, c2 = st.columns(2)
    c1.metric("ML MAE", f"{mae_ml:.2f}")
    c2.metric("ML RMSE", f"{rmse_ml:.2f}")

    fig4, ax4 = plt.subplots()
    ax4.plot(y_test.values[:100], label="Actual")
    ax4.plot(y_pred[:100], label="Predicted")
    ax4.legend()

    st.pyplot(fig4)

except:
    st.warning("LightGBM not available or failed to run.")
