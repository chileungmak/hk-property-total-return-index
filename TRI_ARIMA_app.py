#!/usr/bin/env python
# coding: utf-8


# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="HK Property Forecast", layout="wide")
st.title("Hong Kong Real Estate: Total Return Forecaster")
st.write("Live ARIMA forecasting separating stochastic capital growth from deterministic yields.")

# Cache the data loading so the app doesn't re-read the CSV on every click
@st.cache_data

def load_data():
    url_price = "http://www.rvd.gov.hk/datagovhk/1.4M.csv"
    url_yield = "http://www.rvd.gov.hk/datagovhk/5.1M.csv"
    
    local_price = "1.4M.csv"
    local_yield = "5.1M.csv"
    
    try:
        # 1. Attempt to fetch the live open-data URLs first
        df_price_raw = pd.read_csv(url_price, header=1)
        df_yield_raw = pd.read_csv(url_yield, header=1)
    except Exception:
        # 2. Fallback to local GitHub CSVs if the URLs fail
        st.warning("Live RVD data feed unavailable. Loading backup data from the repository.")
        df_price_raw = pd.read_csv(local_price, header=1)
        df_yield_raw = pd.read_csv(local_yield, header=1)
        
    # 3. Process the data (the logic remains identical regardless of the source)
    df_price = df_price_raw[['Month', 'Class A', 'Class B', 'Class C', 'Class D', 'Class E']].copy()
    df_yield = df_yield_raw[['Month', 'Domestic Class A', 'Domestic Class B', 'Domestic Class C', 'Domestic Class D', 'Domestic Class E']].copy()
    
    df_price.columns = ['Month', 'Price_A', 'Price_B', 'Price_C', 'Price_D', 'Price_E']
    df_yield.columns = ['Month', 'Yield_A', 'Yield_B', 'Yield_C', 'Yield_D', 'Yield_E']
    
    df = pd.merge(df_price, df_yield, on='Month', how='inner')
    df['Date'] = pd.to_datetime(df['Month'], format='%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date']).set_index('Date').drop(columns=['Month'])
    
    return df.apply(pd.to_numeric, errors='coerce').dropna()
df = load_data()

# Interactive dropdown menu
prop_class = st.selectbox("Select Property Class", ['A', 'B', 'C', 'D', 'E'])

# Calculate historical TRI for the selected class
df['Cap_Return'] = df[f'Price_{prop_class}'].pct_change()
df['Inc_Return'] = (df[f'Yield_{prop_class}'] / 100) / 12
df['Total_Return'] = df['Cap_Return'] + df['Inc_Return']
df['Total_Return'] = df['Total_Return'].fillna(0)
df['TRI'] = df[f'Price_{prop_class}'].iloc[0] * (1 + df['Total_Return']).cumprod()

# Fit SARIMAX using our known optimal parameters
train_data = np.log(df[f'Price_{prop_class}'] / df[f'Price_{prop_class}'].shift(1)).dropna()
model = SARIMAX(train_data, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12), enforce_stationarity=False)
results = model.fit(disp=False)

# Forecast the next 12 months
forecast_log = results.get_forecast(steps=12).predicted_mean
forecast_price_ret = np.exp(forecast_log) - 1
assumed_yield = (df[f'Yield_{prop_class}'].iloc[-1] / 100) / 12
forecast_total_ret = forecast_price_ret + assumed_yield

# Build future dates and compound the index
future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
last_tri = df['TRI'].iloc[-1]
forecast_tri = last_tri * (1 + forecast_total_ret).cumprod()

# Render the chart
st.subheader(f"Class {prop_class} Total Return Index Forecast")
fig, ax = plt.subplots(figsize=(12, 6))
plt.style.use('seaborn-v0_8-whitegrid')
ax.plot(df.index, df['TRI'], label='Historical TRI', color='#0275d8', linewidth=2)
ax.plot(future_dates, forecast_tri, label='12-Month Forecast', color='#d9534f', linestyle='--', linewidth=2.5)
ax.fill_between(df.index, df[f'Price_{prop_class}'], df['TRI'], color='#0275d8', alpha=0.1, label='Reinvested Yield')
ax.set_ylabel("Index Value")
ax.legend(loc='upper left')

# Display the plot in the Streamlit app
st.pyplot(fig)


# In[ ]:




