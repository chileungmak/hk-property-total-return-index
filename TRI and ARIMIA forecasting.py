#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# 1. Load the Price Index Data
# Using header=1 tells Pandas to skip the first metadata row and use the actual column names
df_price = pd.read_csv("1.4M.csv", header=1)

# Keep only the Date and the 'By Class' columns
cols_price = ['Month', 'Class A', 'Class B', 'Class C', 'Class D', 'Class E']
df_price = df_price[cols_price].copy()

# Rename columns so they don't clash with the Yield data later
df_price.rename(columns={
    'Class A': 'Price_A', 'Class B': 'Price_B', 'Class C': 'Price_C', 
    'Class D': 'Price_D', 'Class E': 'Price_E'
}, inplace=True)

# 2. Load the Yield Data
df_yield = pd.read_csv("5.1M.csv", header=1)

# Keep only the Date and the 'By Class' columns
cols_yield = ['Month', 'Domestic Class A', 'Domestic Class B', 'Domestic Class C', 'Domestic Class D', 'Domestic Class E']
df_yield = df_yield[cols_yield].copy()

# Rename for clarity
df_yield.rename(columns={
    'Domestic Class A': 'Yield_A', 'Domestic Class B': 'Yield_B', 'Domestic Class C': 'Yield_C', 
    'Domestic Class D': 'Yield_D', 'Domestic Class E': 'Yield_E'
}, inplace=True)

# 3. Merge and Clean
# Inner merge ensures we only keep months where BOTH price and yield exist
df_merged = pd.merge(df_price, df_yield, on='Month', how='inner')

# Convert the "MM-YYYY" string into a proper Datetime index
df_merged['Date'] = pd.to_datetime(df_merged['Month'], format='%m-%Y', errors='coerce')
df_merged = df_merged.dropna(subset=['Date'])
df_merged.set_index('Date', inplace=True)
df_merged.drop(columns=['Month'], inplace=True)

# Ensure all data is numeric (coerces any random asterisks/footnotes to NaN)
df_merged = df_merged.apply(pd.to_numeric, errors='coerce')

print("Master DataFrame Built Successfully.")
print(df_merged.head())


# In[2]:


# List of property classes based on RVD categorization
classes = ['A', 'B', 'C', 'D', 'E']

for cls in classes:
    # 1. Capital Return (Percentage change in Price Index)
    df_merged[f'Cap_Return_{cls}'] = df_merged[f'Price_{cls}'].pct_change()
    
    # 2. Income Return (Convert annualized yield % to a monthly decimal)
    df_merged[f'Inc_Return_{cls}'] = (df_merged[f'Yield_{cls}'] / 100) / 12
    
    # 3. Total Monthly Return
    df_merged[f'TR_{cls}'] = df_merged[f'Cap_Return_{cls}'] + df_merged[f'Inc_Return_{cls}']
    
    # Replace the first month's NaN with 0 to set the base accurately
    df_merged[f'TR_{cls}'] = df_merged[f'TR_{cls}'].fillna(0)
    
    # 4. Calculate Total Return Index (TRI)
    base_price = df_merged[f'Price_{cls}'].iloc[0]
    df_merged[f'TRI_{cls}'] = base_price * (1 + df_merged[f'TR_{cls}']).cumprod()

# Print the comparison between raw price and TRI for Class A (Small units) and Class E (Luxury units)
print(df_merged[['Price_A', 'TRI_A', 'Price_E', 'TRI_E']].tail())


# In[3]:


import matplotlib.pyplot as plt

# Set up the plot aesthetics
plt.figure(figsize=(12, 6))
plt.style.use('seaborn-v0_8-whitegrid') # Gives a clean, professional background

# Plot the standard Price Index (Dashed line to show it's the "baseline")
plt.plot(df_merged.index, df_merged['Price_A'], 
         label='Class A Price Index (Capital Value Only)', 
         color='#d9534f', linestyle='--', linewidth=2)

# Plot the Total Return Index (Solid, bold line to emphasize the true return)
plt.plot(df_merged.index, df_merged['TRI_A'], 
         label='Class A Total Return Index (Reinvested Rent)', 
         color='#0275d8', linewidth=2.5)

# Add titles and labels
plt.title('Hong Kong Class A Residential: Capital Value vs. Total Return', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Index Value (Jan 1999 Base)', fontsize=12)

# Fill the area between the lines to visually represent the accumulated rental yield
plt.fill_between(df_merged.index, df_merged['Price_A'], df_merged['TRI_A'], color='#0275d8', alpha=0.1)

plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()

# Save the image so you can upload it to GitHub
plt.savefig('total_return_divergence_classA.png', dpi=300)
print("Chart saved as 'total_return_divergence_classA.png'")

# Display it on screen
plt.show()


# In[4]:


from statsmodels.tsa.stattools import adfuller
import numpy as np

print("\n--- ARIMA PREPARATION: STATIONARITY TEST ---")

# 1. Calculate Log Returns for Class A Price Index
# Log returns are standard in quantitative finance as they are time-additive
df_merged['Log_Return_A'] = np.log(df_merged['Price_A'] / df_merged['Price_A'].shift(1))

# Drop the first NaN row created by the shift
train_data = df_merged['Log_Return_A'].dropna()

# 2. Run the Augmented Dickey-Fuller (ADF) Test
# This tests the null hypothesis that a unit root is present (data is non-stationary)
adf_result = adfuller(train_data)

print(f"ADF Statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

# 3. Interpret the result
if adf_result[1] < 0.05:
    print("Conclusion: The Log Return data is STATIONARY (p-value < 0.05).")
    print("We can proceed with an ARIMA(p, 0, q) model.")
else:
    print("Conclusion: The Log Return data is NON-STATIONARY (p-value >= 0.05).")
    print("We will need to difference the data further, using an ARIMA(p, 1, q) model.")


# In[5]:


import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Create a figure with two side-by-side subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot ACF to identify the 'q' parameter (Moving Average)
plot_acf(train_data, ax=axes[0], lags=24)
axes[0].set_title('Autocorrelation (ACF) - Determines $q$')
axes[0].set_xlabel('Lags (Months)')
axes[0].set_ylabel('Correlation')

# Plot PACF to identify the 'p' parameter (AutoRegressive)
# 'ywm' method is the standard solver for PACF calculations
plot_pacf(train_data, ax=axes[1], lags=24, method='ywm')
axes[1].set_title('Partial Autocorrelation (PACF) - Determines $p$')
axes[1].set_xlabel('Lags (Months)')
axes[1].set_ylabel('Correlation')

plt.tight_layout()
plt.savefig('acf_pacf_plots.png', dpi=300)
print("ACF and PACF charts saved as 'acf_pacf_plots.png'")
plt.show()


# In[6]:


import pmdarima as pm

print("\n--- RUNNING AUTO-ARIMA GRID SEARCH ---")

# auto_arima systematically searches for the best model parameters
auto_model = pm.auto_arima(train_data, 
                           start_p=0, start_q=0,
                           max_p=3, max_q=3, 
                           m=12,              # Checks for 12-month seasonality
                           start_P=0, seasonal=True,
                           d=0, D=0,          # We know d=0 from your ADF test
                           trace=True,        # Shows the search process in the console
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

print("\n=== OPTIMAL ARIMA MODEL SUMMARY ===")
print(auto_model.summary())


# In[7]:


# 1. Forecast the next 12 months of Log Returns
n_periods = 12
forecast_log_returns, conf_int = auto_model.predict(n_periods=n_periods, return_conf_int=True)

# 2. Convert Log Returns back to Simple Price Returns
# Formula: R = exp(r) - 1
forecast_price_returns = np.exp(forecast_log_returns) - 1

# 3. Assume Yield Remains Constant
# We take the most recent annualized yield, convert to decimal, and divide by 12
latest_yield_annual = df_merged['Yield_A'].iloc[-1]
monthly_yield_assumption = (latest_yield_annual / 100) / 12

# 4. Calculate Future Total Monthly Returns
forecast_total_returns = forecast_price_returns + monthly_yield_assumption

# 5. Build the Forecasted Total Return Index (TRI)
# Start compounding from the very last known TRI value in our historical dataset
last_known_tri = df_merged['TRI_A'].iloc[-1]
forecast_tri = last_known_tri * (1 + forecast_total_returns).cumprod()

# 6. Create Future Dates for the Index
last_date = df_merged.index[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_periods, freq='MS')

# Format as a clean DataFrame
df_forecast = pd.DataFrame({
    'Forecasted_Price_Return': forecast_price_returns,
    'Assumed_Monthly_Yield': monthly_yield_assumption,
    'Forecasted_TRI_Class_A': forecast_tri
}, index=future_dates)

print("\n=== 12-MONTH TOTAL RETURN INDEX FORECAST ===")
print(df_forecast)


# In[ ]:




