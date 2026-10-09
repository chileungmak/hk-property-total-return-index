import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
        
        try:
            ar_model = SARIMAX(train_series, order=(1,0,0), seasonal_order=(1,0,0,12), enforce_stationarity=False)
            ar_res = ar_model.fit(disp=False)
            ar_forecast = ar_res.forecast(steps=horizon).sum()
            ar1_errors.append(ar_forecast - actual_cumulative)
        except Exception:
            ar1_errors.append(np.nan)
            
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
    st.write("Live quantitative forecasting, cross-sectional risk analytics, and market factor decomposition.")
    
    df_raw = load_data()

    classes = ['A', 'B', 'C', 'D', 'E']
    tri_dfs = {}
    
    for c in classes:
        class_df = df_raw[[f'Price_{c}', f'Yield_{c}']].dropna().copy()
        tri_data = calculate_historical_tri(class_df[f'Price_{c}'], class_df[f'Yield_{c}'])
        class_df['Cap_Return'] = tri_data['Cap_Return']
        class_df['Inc_Return'] = tri_data['Inc_Return']
        class_df['Total_Return'] = tri_data['Total_Return']
        class_df['TRI'] = tri_data['TRI']
        tri_dfs[c] = class_df

    tab1, tab2, tab3 = st.tabs(["🔮 Forecasting", "⚖️ Risk & Return", "🧬 Factor Analysis"])

    with tab1:
        st.header("Forecasting & Diagnostics")
        prop_class = st.selectbox("Select Property Class", classes)
        
        df = tri_dfs[prop_class]
        train_data = np.log(df[f'Price_{prop_class}'] / df[f'Price_{prop_class}'].shift(1)).dropna()

        # 1. ARIMA Baseline
        ar_model_full = SARIMAX(train_data, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12), enforce_stationarity=False)
        ar_res_full = ar_model_full.fit(disp=False)
        ar_forecast_res = ar_res_full.get_forecast(steps=12)
        ar_forecast_log = ar_forecast_res.predicted_mean
        ar_conf_int = ar_forecast_res.conf_int(alpha=0.05) 

        # 2. XGBoost
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

        fig.update_layout(xaxis_title="Date", yaxis_title="Index Value", template="plotly_white", legend=dict(x=0.01, y=0.99), margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Model Diagnostics (Walk-Forward Validation)")
        st.write("Evaluating the 12-month direct forecast accuracy over the last 24 test periods.")
        with st.spinner("Running walk-forward validation..."):
            ar_rmse, ar_mae, xgb_rmse, xgb_mae = run_walk_forward_validation(train_data, test_size=24, horizon=12, lags=13)

        metrics_df_val = pd.DataFrame({
            "Metric": ["RMSE (12-Month Log Return)", "MAE (12-Month Log Return)"],
            "ARIMA(1,0,0)": [round(ar_rmse, 4), round(ar_mae, 4)],
            "XGBoost (Direct)": [round(xgb_rmse, 4), round(xgb_mae, 4)]
        })
        st.dataframe(metrics_df_val, hide_index=True)

        if xgb_rmse < ar_rmse:
            st.success("XGBoost outperformed ARIMA in 12-month horizon forecasting during the testing period.")
        else:
            st.info("ARIMA outperformed XGBoost in 12-month horizon forecasting during the testing period.")

    with tab2:
        st.header("Risk & Return Analytics")
        st.write("""
        **Methodology:** 
        This section evaluates the cross-sectional risk and return profile of each property class. 
        - **Annualized Return & Volatility:** Scaled from monthly log returns.
        - **Sharpe Ratio:** Measures return per unit of total risk (assuming 0% risk-free rate for baseline).
        - **Sortino Ratio:** Measures return relative strictly to *downside* volatility (penalizing only negative returns).
        - **Max Drawdown:** The largest historical peak-to-trough drop in the Total Return Index.
        """)
        
        metrics = []
        for c in classes:
            cdf = tri_dfs[c]
            ret = cdf['Total_Return']
            
            start_date = cdf.index.min().strftime('%Y-%m')
            end_date = cdf.index.max().strftime('%Y-%m')
            
            ann_ret = ret.mean() * 12
            ann_vol = ret.std() * np.sqrt(12)
            sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
            
            downside_var = np.mean(np.minimum(0, ret)**2)
            downside_vol = np.sqrt(downside_var) * np.sqrt(12)
            sortino = ann_ret / downside_vol if downside_vol != 0 else np.nan
            
            roll_max = cdf['TRI'].cummax()
            drawdown = cdf['TRI'] / roll_max - 1.0
            max_dd = drawdown.min()
            
            metrics.append({
                "Property Class": f"Class {c}",
                "Start Date": start_date,
                "End Date": end_date,
                "Ann. Return": ann_ret,
                "Ann. Volatility": ann_vol,
                "Sharpe Ratio": sharpe,
                "Sortino Ratio": sortino,
                "Max Drawdown": max_dd
            })
            
        metrics_df = pd.DataFrame(metrics)
        st.dataframe(
            metrics_df.style.format({
                "Ann. Return": "{:.2%}", 
                "Ann. Volatility": "{:.2%}", 
                "Sharpe Ratio": "{:.2f}", 
                "Sortino Ratio": "{:.2f}", 
                "Max Drawdown": "{:.2%}"
            }),
            hide_index=True,
            use_container_width=True
        )
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=metrics_df["Ann. Volatility"], 
            y=metrics_df["Ann. Return"], 
            mode='markers+text',
            text=metrics_df["Property Class"],
            textposition="top center",
            marker=dict(size=14, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ))
        fig2.update_layout(
            title="Risk vs Reward Profile", 
            xaxis_title="Annualized Volatility", 
            yaxis_title="Annualized Return", 
            template="plotly_white",
            xaxis=dict(tickformat=".1%", rangemode="tozero"),
            yaxis=dict(tickformat=".1%", rangemode="tozero")
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.header("Factor Analysis (PCA)")
        st.write("""
        **Methodology:**
        Principal Component Analysis (PCA) decomposes the monthly returns of all five property classes into uncorrelated underlying factors. By standardizing the returns and extracting their eigen-components, PCA mathematically isolates the true drivers of the Hong Kong real estate market.
        """)
        
        pca_df = pd.DataFrame({f"Class {c}": tri_dfs[c]['Total_Return'] for c in classes}).dropna()
        
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(pca_df)
        
        pca = PCA()
        pca.fit(scaled_returns)
        
        explained_variance = pca.explained_variance_ratio_
        
        # Align PCA directions for intuitive display
        # Force PC1 to be universally positive (Market Factor)
        pc1_loadings = pca.components_[0]
        if pc1_loadings[0] < 0:
            pc1_loadings = -pc1_loadings
            
        # Force PC2 to be positive for Class E (Luxury Divergence)
        pc2_loadings = pca.components_[1]
        if pc2_loadings[-1] < 0:
            pc2_loadings = -pc2_loadings
            
        col1, col2 = st.columns(2)
        
        with col1:
            fig3a = go.Figure(data=[
                go.Bar(
                    x=[f"PC{i+1}" for i in range(len(classes))], 
                    y=explained_variance,
                    marker_color='#0275d8'
                )
            ])
            fig3a.update_layout(
                title="Explained Variance (Scree Plot)", 
                yaxis_title="% of Variance Explained", 
                yaxis_tickformat=".1%", 
                template="plotly_white"
            )
            st.plotly_chart(fig3a, use_container_width=True)
            
        with col2:
            fig3b = go.Figure(data=[
                go.Bar(name='PC1 (Market Factor)', x=[f"Class {c}" for c in classes], y=pc1_loadings, marker_color='#5cb85c'),
                go.Bar(name='PC2 (Divergence)', x=[f"Class {c}" for c in classes], y=pc2_loadings, marker_color='#d9534f')
            ])
            fig3b.update_layout(
                title="Factor Loadings (PC1 vs PC2)", 
                yaxis_title="Loading Weight", 
                barmode='group', 
                template="plotly_white"
            )
            st.plotly_chart(fig3b, use_container_width=True)
        
        st.info(
            "**Interpretation:** PC1 is the **wider property market factor**—the macro beta driving the entire Hong Kong real estate sector. "
            "PC2 highlights structural market divergence. In Hong Kong, Classes A to C are small to medium size property which is usually a proxy for mass property, while Classes D and E are large size apartments which are considered a proxy for luxury. The opposing loadings on PC2 specifically isolate the behavioral difference between these mass and luxury segments."
        )

        st.divider()
        st.subheader("Factor Dynamics Over Time")
        st.write("Does the mass market consistently outperform luxury? Has the exposure to these factors remained stable during policy changes (e.g., Stamp Duties)?")
        
        # Calculate actual portfolio returns for the factors
        pc1_weights_norm = pc1_loadings / np.sum(np.abs(pc1_loadings))
        pc2_weights_norm = pc2_loadings / np.sum(np.abs(pc2_loadings))
        
        pc1_port_ret = pca_df.dot(pc1_weights_norm)
        pc2_port_ret = pca_df.dot(pc2_weights_norm)
        
        # Start indices at 100
        cum_pc1 = (1 + pc1_port_ret).cumprod() * 100
        cum_pc2 = (1 + pc2_port_ret).cumprod() * 100
        
        col3, col4 = st.columns(2)
        
        with col3:
            fig3c = go.Figure()
            fig3c.add_trace(go.Scatter(x=pca_df.index, y=cum_pc1, mode='lines', name='PC1 (Market Beta)', line=dict(color='#5cb85c')))
            fig3c.add_trace(go.Scatter(x=pca_df.index, y=cum_pc2, mode='lines', name='PC2 (PCA-Weighted Long/Short Portfolio)', line=dict(color='#d9534f')))
            fig3c.update_layout(
                title="Cumulative Factor Returns (Base=100)", 
                xaxis_title="Date", 
                yaxis_title="Index Value", 
                template="plotly_white",
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig3c, use_container_width=True)
            
        with col4:
            # Shortened window to 36 months to properly map historical structural breaks
            rolling_window = 36
            dates = []
            pc1_a, pc1_e, pc2_a, pc2_e = [], [], [], []
            
            for i in range(rolling_window, len(pca_df)):
                window_data = pca_df.iloc[i-rolling_window:i]
                window_scaled = StandardScaler().fit_transform(window_data)
                
                roll_pca = PCA(n_components=2)
                roll_pca.fit(window_scaled)
                
                w1 = roll_pca.components_[0]
                if w1[0] < 0: w1 = -w1
                    
                w2 = roll_pca.components_[1]
                if w2[-1] < 0: w2 = -w2
                    
                dates.append(pca_df.index[i])
                pc1_a.append(w1[0]) 
                pc1_e.append(w1[4]) 
                pc2_a.append(w2[0])
                pc2_e.append(w2[4])
                
            fig3d = go.Figure()
            fig3d.add_trace(go.Scatter(x=dates, y=pc1_a, mode='lines', name='Class A (Mass) on PC1', line=dict(color='#1f77b4', dash='solid')))
            fig3d.add_trace(go.Scatter(x=dates, y=pc1_e, mode='lines', name='Class E (Luxury) on PC1', line=dict(color='#9467bd', dash='solid')))
            fig3d.add_trace(go.Scatter(x=dates, y=pc2_a, mode='lines', name='Class A (Mass) on PC2', line=dict(color='#1f77b4', dash='dot')))
            fig3d.add_trace(go.Scatter(x=dates, y=pc2_e, mode='lines', name='Class E (Luxury) on PC2', line=dict(color='#9467bd', dash='dot')))
            
            # Add Vertical Reference Lines tailored to trailing-window logic
            fig3d.add_vline(x="2012-10-26", line_width=1.5, line_dash="dash", line_color="black", annotation_text="BSD Introduced (Oct 2012)", annotation_position="top left")
            fig3d.add_vline(x="2015-10-26", line_width=1, line_dash="dot", line_color="gray", annotation_text="36M Post-BSD (Collapse)", annotation_position="bottom right")
            fig3d.add_vline(x="2018-07-06", line_width=1.5, line_dash="dash", line_color="black", annotation_text="Trade War Escalation", annotation_position="top left")
            fig3d.add_vline(x="2019-06-09", line_width=1.5, line_dash="dash", line_color="gray", annotation_text="Unrest", annotation_position="bottom right")
            
            fig3d.update_layout(
                title="36-Month Rolling Factor Exposures", 
                xaxis_title="Date", 
                yaxis_title="Loading Weight", 
                template="plotly_white",
                legend=dict(x=0.01, y=0.5)
            )
            st.plotly_chart(fig3d, use_container_width=True)

if __name__ == "__main__":
    main()
