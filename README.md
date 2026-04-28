# Coffee Demand Forecasting Dashboard

## Overview

This project presents an end-to-end data analysis and forecasting solution for a coffee retail business. The goal is to analyze transaction data, identify demand patterns, and forecast future demand to support better decision-making.

The application is built using Streamlit and provides an interactive dashboard for exploring key metrics, trends, and forecasts.

---

## Objectives

- Analyze sales and transaction data
- Identify peak demand hours and days
- Visualize store-level performance
- Forecast future demand using time-series modeling
- Provide insights for business optimization

---

## Features

### Data Integration
- Loads dataset directly from GitHub
- Supports user-uploaded Excel files

### Data Processing
- Cleans and preprocesses transaction data
- Extracts time-based features (hour, day)
- Constructs a usable time series

### Key Metrics
- Total Revenue
- Total Transactions
- Average Order Value
- Peak Hour Detection

### Visual Analytics
- Daily revenue trends
- Peak hour distribution
- Store vs hour demand heatmap
- Peak demand day identification

### Forecasting
- Uses Prophet for time-series forecasting
- Displays predicted demand trends
- Includes confidence intervals

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Prophet

---

## Project Structure
Afficionado_Coffee_Streamlit/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── README.md
│
└── screenshots/
├── README.md
├── dashboard.png
├── forecast.png
├── heatmap.png
├── peak_hour.png

---

## Installation & Setup

1. Clone the repository:
git clone https://github.com/raaahuuull/Afficionado_Coffee_Streamlit.git
cd Afficionado_Coffee_Streamlit

2. Install dependencies:

pip install -r requirements.txt


3. Run the application: streamlit run app.py
Streamlit link- https://afficionadocoffeeapp-vacpn4ustrbdh5cfkzxxyh.streamlit.app/

---

## Usage

- Use sidebar to load dataset (GitHub or upload)
- Select store for forecasting
- Adjust forecast duration
- Explore dashboard insights

---

## Key Insights

- Demand follows strong time-based patterns
- Peak hours indicate customer behavior trends
- Store performance can be compared visually
- Forecasting helps in planning inventory and staffing

---

## Conclusion

This project demonstrates how data analytics and time-series forecasting can be combined to extract meaningful business insights. The dashboard enables intuitive exploration and supports data-driven decisions.

---

## Future Improvements

- Add advanced models (LSTM, XGBoost)
- Improve UI with more interactive filters
- Add report download feature
- Enhance forecasting accuracy

---

## Author

Rahul Raj
