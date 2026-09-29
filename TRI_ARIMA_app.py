import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import warnings

warnings.filterwarnings("ignore")

@st.cache_data
def load_data():
    url_price = "http://www.rvd.gov.hk/datagovhk/1.4M.csv"
    url_yield = "http://www.rvd.gov.hk/datagovhk/5.1M.csv"
    
    local_price = "1.4M.csv"
    local_yield = "5.1M.csv"
    
    try:
        df_price_raw = pd.read_csv(url_price, header=1)
        df_yield_raw = pd.read_csv(url_yield, header=1)
    except Exception:
        st.warning("Live RVD data feed unavailable. Loading backup data from the repository.")
        df_price_raw = pd.read_csv(local_price, header=1)
        df_yield_raw = pd.read_csv(local_yield, header=1)
        
    df_price = df_price_raw[['Month', 'Class A', 'Class B', 'Class C', 'Class D', 'Class E']].copy()
    df_yield = df_yield_raw[['Month', 'Domestic Class A', 'Domestic Class B', 'Domestic Class C', 'Domestic Class D', 'Domestic Class E']].copy()
    
    df_price.columns = ['Month', 'Price_A', 'Price_B', 'Price_C', 'Price_D', 'Price_E']
    df_yield.columns = ['Month', 'Yield_A', 'Yield_B', 'Yield_C', 'Yield_D', 'Yield_E']
    
    df = pd.merge(df_price, df_yield, on='Month', how='inner')
    df['Date'] = pd.to_datetime(df['Month'], format='%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date']).set_index('Date').drop(columns=['Month'])
    
    return df.apply(pd.to_numeric, errors='coerce')

def calculate_historical_tri(price_col, yield_col):
    """
    Calculates the Total Return Index (TRI) mathematically by combining 
    stochastic capital returns with deterministic yield returns.
    """
    cap_return = price_col.pct_change()
    inc_return = (yield_col / 100) / 12
    total_return = cap_return + inc_return
    total_return = total_return.fillna(0)
    
    tri = price_col.iloc[0] * (1 + total_return).cumprod()
    
    return pd.DataFrame({
        'Cap_Return': cap_return,
        'Inc_Return': inc_return,
        'Total_Return': total_return,
        'TRI': tri
    })

def prepare_ml_data(series, lags=13, horizon=12):
    df = pd.DataFrame({'y': series})
    
    feature_cols = []
    for i in range(lags):
        col_name = f'lag_{i}'
        df[col_name] = df['y'].shift(i)
        feature_cols.append(col_name)
        
    target_cols = []
    for h in range(1, horizon + 1):
        col_name = f'target_{h}'
        df[col_name] = df['y'].shift(-h)
        target_cols.append(col_name)
        
    df = df.dropna()
    X = df[feature_cols]
    Y = df[target_cols]
    return X, Y

@st.cache_data
def run_walk_forward_validation(series, test_size=24, horizon=12, lags=13):
    ar1_errors = []
    xgb_errors = []
    
    n = len(series)
    start_test = n - horizon - test_size
    
    X_full, Y_full = prepare_ml_data(series, lags=lags, horizon=horizon)
    
    for T_idx in range(start_test, n - horizon):
        train_series = series.iloc[:T_idx+1]
        actual_cumulative = series.iloc[T_idx+1 : T_idx+1+horizon].sum()
        
        # AR1 (SARIMAX)
        try:
            ar_model = SARIMAX(train_series, order=(1,0,0), seasonal_order=(1,0,0,12), enforce_stationarity=False)
            ar_res = ar_model.fit(disp=False)
            ar_forecast = ar_res.forecast(steps=horizon).sum()
            ar1_errors.append(ar_forecast - actual_cumulative)
        except Exception:
            ar1_errors.append(np.nan)
            
        # XGBoost (Direct)
        train_end_date = series.index[T_idx - horizon]
        train_mask = (X_full.index <= train_end_date)
        X_train = X_full[train_mask]
        Y_train = Y_full[train_mask]
        
        xgb = MultiOutputRegressor(XGBRegressor(n_estimators=50, max_depth=3, random_state=42, objective='reg:squarederror'))
        if len(X_train) > 0:
            xgb.fit(X_train, Y_train)
            x_pred = pd.DataFrame([series.iloc[T_idx-lags+1 : T_idx+1].values[::-1]], columns=[f'lag_{i}' for i in range(lags)])
            xgb_forecast = xgb.predict(x_pred)[0].sum()
            xgb_errors.append(xgb_forecast - actual_cumulative)
        else:
            xgb_errors.append(np.nan)
            
    ar1_rmse = np.sqrt(np.nanmean(np.square(ar1_errors)))
    ar1_mae = np.nanmean(np.abs(ar1_errors))
    xgb_rmse = np.sqrt(np.nanmean(np.square(xgb_errors)))
    xgb_mae = np.nanmean(np.abs(xgb_errors))
    
    return ar1_rmse, ar1_mae, xgb_rmse, xgb_mae


def main():
    st.set_page_config(page_title="HK Property Forecast", layout="wide")
    st.title("Hong Kong Real Estate: Total Return Forecaster")
    st.write("Live ARIMA & XGBoost forecasting separating stochastic capital growth from deterministic yields.")
    
    df_raw = load_data()

    prop_class = st.selectbox("Select Property Class", ['A', 'B', 'C', 'D', 'E'])

    df = df_raw[[f'Price_{prop_class}', f'Yield_{prop_class}']].dropna().copy()

    # Calculate historical TRI using the refactored, testable function
    tri_data = calculate_historical_tri(df[f'Price_{prop_class}'], df[f'Yield_{prop_class}'])
    df['Cap_Return'] = tri_data['Cap_Return']
    df['Inc_Return'] = tri_data['Inc_Return']
    df['Total_Return'] = tri_data['Total_Return']
    df['TRI'] = tri_data['TRI']

    train_data = np.log(df[f'Price_{prop_class}'] / df[f'Price_{prop_class}'].shift(1)).dropna()

    # 1. ARIMA Baseline (with 95% Confidence Intervals)
    ar_model_full = SARIMAX(train_data, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12), enforce_stationarity=False)
    ar_res_full = ar_model_full.fit(disp=False)
    ar_forecast_res = ar_res_full.get_forecast(steps=12)
    ar_forecast_log = ar_forecast_res.predicted_mean
    ar_conf_int = ar_forecast_res.conf_int(alpha=0.05) 

    # 2. XGBoost Direct Multi-Step
    lags = 13
    X_full, Y_full = prepare_ml_data(train_data, lags=lags, horizon=12)
    xgb_model_full = MultiOutputRegressor(XGBRegressor(n_estimators=50, max_depth=3, random_state=42, objective='reg:squarederror'))
    xgb_model_full.fit(X_full, Y_full)
    x_future = pd.DataFrame([train_data.iloc[-lags:].values[::-1]], columns=[f'lag_{i}' for i in range(lags)])
    xgb_forecast_log = pd.Series(xgb_model_full.predict(x_future)[0], index=ar_forecast_log.index)

    # 3. Combine with Yield 
    assumed_yield = (df[f'Yield_{prop_class}'].iloc[-12:].mean() / 100) / 12

    ar_total_ret = (np.exp(ar_forecast_log) - 1) + assumed_yield
    xgb_total_ret = (np.exp(xgb_forecast_log) - 1) + assumed_yield

    ar_lower_ret = (np.exp(ar_conf_int.iloc[:, 0]) - 1) + assumed_yield
    ar_upper_ret = (np.exp(ar_conf_int.iloc[:, 1]) - 1) + assumed_yield

    future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='MS')
    last_tri = df['TRI'].iloc[-1]

    ar_forecast_tri = last_tri * (1 + ar_total_ret).cumprod()
    xgb_forecast_tri = last_tri * (1 + xgb_total_ret).cumprod()

    ar_lower_tri = last_tri * (1 + ar_lower_ret).cumprod()
    ar_upper_tri = last_tri * (1 + ar_upper_ret).cumprod()

    plot_dates = pd.concat([pd.Series([df.index[-1]]), pd.Series(future_dates)])
    plot_ar_tri = pd.concat([pd.Series([last_tri]), pd.Series(ar_forecast_tri)])
    plot_xgb_tri = pd.concat([pd.Series([last_tri]), pd.Series(xgb_forecast_tri)])
    plot_ar_lower = pd.concat([pd.Series([last_tri]), pd.Series(ar_lower_tri)])
    plot_ar_upper = pd.concat([pd.Series([last_tri]), pd.Series(ar_upper_tri)])

    st.subheader(f"Class {prop_class} Total Return Index Forecast")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pd.concat([plot_dates, plot_dates[::-1]]),
        y=pd.concat([plot_ar_upper, plot_ar_lower[::-1]]),
        fill='toself',
        fillcolor='rgba(217, 83, 79, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='ARIMA 95% CI'
    ))

    fig.add_trace(go.Scatter(x=df.index, y=df['TRI'], mode='lines', name='Historical TRI', line=dict(color='#0275d8', width=2)))
    fig.add_trace(go.Scatter(x=plot_dates, y=plot_ar_tri, mode='lines', name='ARIMA Forecast', line=dict(color='#d9534f', width=2.5, dash='dash')))
    fig.add_trace(go.Scatter(x=plot_dates, y=plot_xgb_tri, mode='lines', name='XGBoost Forecast', line=dict(color='#5cb85c', width=2.5, dash='dot')))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Index Value",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Diagnostics (Walk-Forward Validation)")
    st.write("Evaluating the 12-month direct forecast accuracy over the last 24 test periods.")
    st.caption("Note: XGBoost is configured with 13 lags to match ARIMA's 12-month seasonality. Yields are fixed forward using a trailing 12-month average.")

    with st.spinner("Running walk-forward validation..."):
        ar_rmse, ar_mae, xgb_rmse, xgb_mae = run_walk_forward_validation(train_data, test_size=24, horizon=12, lags=13)

    metrics_df = pd.DataFrame({
        "Metric": ["RMSE (12-Month Log Return)", "MAE (12-Month Log Return)"],
        "ARIMA(1,0,0)": [round(ar_rmse, 4), round(ar_mae, 4)],
        "XGBoost (Direct)": [round(xgb_rmse, 4), round(xgb_mae, 4)]
    })
    st.dataframe(metrics_df, hide_index=True)

    if xgb_rmse < ar_rmse:
        st.success("XGBoost outperformed ARIMA in 12-month horizon forecasting during the testing period.")
    else:
        st.info("ARIMA outperformed XGBoost in 12-month horizon forecasting during the testing period.")

if __name__ == "__main__":
    main()
