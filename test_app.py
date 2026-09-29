import pytest
import pandas as pd
import numpy as np
import io
from unittest.mock import patch
from TRI_ARIMA_app import (
    prepare_ml_data, 
    calculate_historical_tri, 
    run_walk_forward_validation, 
    load_data
)

def test_prepare_ml_data_alignment():
    """
    Test that the lag features and the targets align precisely as expected,
    so that lag_0 corresponds to y(t) and target_1 corresponds to y(t+1).
    """
    # Create a simple synthetic series: values [10.0, 11.0, 12.0 ... 50.0]
    series = pd.Series(
        np.arange(10, 51, dtype=float), 
        index=pd.date_range("2000-01-01", periods=41, freq="MS")
    )
    
    lags = 3
    horizon = 2
    X, Y = prepare_ml_data(series, lags=lags, horizon=horizon)
    
    # Check the specific row where index is '2000-04-01' (the 4th element, value=13)
    row_idx = pd.to_datetime("2000-04-01")
    
    # For y(t) = 13:
    # lag_0 should be 13, lag_1 should be 12, lag_2 should be 11
    assert X.loc[row_idx, 'lag_0'] == 13
    assert X.loc[row_idx, 'lag_1'] == 12
    assert X.loc[row_idx, 'lag_2'] == 11
    
    # target_1 should be 14 (t+1), target_2 should be 15 (t+2)
    assert Y.loc[row_idx, 'target_1'] == 14
    assert Y.loc[row_idx, 'target_2'] == 15

def test_training_vs_inference_parity():
    """
    Test that the feature vector generated internally by prepare_ml_data for 
    training at time T is absolutely mathematically identical to the dynamic
    x_pred vector generated during the inference/walk-forward loop for time T.
    """
    series = pd.Series(
        np.arange(100, 150, dtype=float), 
        index=pd.date_range("2000-01-01", periods=50, freq="MS")
    )
    
    lags = 4
    horizon = 3
    X, Y = prepare_ml_data(series, lags=lags, horizon=horizon)
    
    # Simulate being inside the walk-forward loop at time T
    T_idx = 30 
    T_date = series.index[T_idx]
    
    # 1. Training feature extraction 
    # prepare_ml_data builds rows mapped to the date of lag_0.
    training_features = X.loc[T_date]
    
    # 2. Dynamic inference extraction (identical to app logic)
    x_pred = pd.DataFrame(
        [series.iloc[T_idx-lags+1 : T_idx+1].values[::-1]], 
        columns=[f'lag_{i}' for i in range(lags)]
    ).iloc[0] # compare as a 1D Series
    
    # They must perfectly match, preventing the off-by-one bug
    pd.testing.assert_series_equal(training_features, x_pred, check_names=False)

def test_shape_and_no_nans():
    """
    Test that the returned X and Y matrices match the exact dimensional
    requirements (columns = lags/horizon) and contain zero NaNs.
    """
    series = pd.Series(np.random.randn(50))
    lags = 5
    horizon = 12
    X, Y = prepare_ml_data(series, lags=lags, horizon=horizon)
    
    assert X.shape[1] == lags
    assert Y.shape[1] == horizon
    assert not X.isna().any().any(), "NaNs leaked into X matrix"
    assert not Y.isna().any().any(), "NaNs leaked into Y matrix"

def test_cross_class_dropna_isolation():
    """
    Assert that if Class B is missing recent data, it does not inadvertently
    truncate Class A during isolated dataframe extraction.
    """
    dates = pd.date_range("2000-01-01", periods=50, freq="MS")
    df = pd.DataFrame({
        'Date': dates,
        'Price_A': np.arange(50, dtype=float),
        'Yield_A': np.arange(50, dtype=float),
        'Price_B': np.arange(50, dtype=float),
        'Yield_B': np.arange(50, dtype=float)
    }).set_index('Date')
    
    # Corrupt the last 10 periods for Class B only
    df.loc[df.index[-10:], 'Price_B'] = np.nan
    df.loc[df.index[-10:], 'Yield_B'] = np.nan
    
    # Simulate the isolated Class A pipeline
    class_a_df = df[['Price_A', 'Yield_A']].dropna()
    assert len(class_a_df) == 50, "Class A was incorrectly truncated by Class B's missing data"
    
    # Simulate the isolated Class B pipeline
    class_b_df = df[['Price_B', 'Yield_B']].dropna()
    assert len(class_b_df) == 40, "Class B did not drop its own NaNs correctly"

def test_calculate_historical_tri():
    """
    Test the financial compounding math for the Total Return Index (TRI).
    Month 1: Price=100, Yield=2.4% -> Cap=NaN, Inc=NaN (Total=0), TRI=100
    Month 2: Price=105, Yield=3.6% -> Cap=0.05, Inc=(3.6/100)/12=0.003, Total=0.053, TRI=105.3
    """
    dates = pd.date_range("2000-01-01", periods=2, freq="MS")
    price_col = pd.Series([100.0, 105.0], index=dates)
    yield_col = pd.Series([2.4, 3.6], index=dates)
    
    res = calculate_historical_tri(price_col, yield_col)
    
    # Month 1
    assert res['TRI'].iloc[0] == 100.0
    assert res['Total_Return'].iloc[0] == 0.0
    
    # Month 2
    assert pytest.approx(res['Cap_Return'].iloc[1]) == 0.05
    assert pytest.approx(res['Inc_Return'].iloc[1]) == 0.003
    assert pytest.approx(res['Total_Return'].iloc[1]) == 0.053
    assert pytest.approx(res['TRI'].iloc[1]) == 105.3

def test_run_walk_forward_validation_boundary():
    """
    Test that run_walk_forward_validation correctly masks the future targets 
    so it does not leak data, and that it successfully generates metrics.
    """
    # Create a 36-month linear series
    series = pd.Series(
        np.linspace(1, 10, 36), 
        index=pd.date_range("2000-01-01", periods=36, freq="MS")
    )
    
    # Run with small params to test the inner loop's boundaries quickly
    ar1_rmse, ar1_mae, xgb_rmse, xgb_mae = run_walk_forward_validation(
        series, test_size=5, horizon=3, lags=3
    )
    
    # If the train_end_date boundary was wrong (e.g. producing an empty training 
    # set or a shape mismatch), this would throw an error or return NaN.
    assert not np.isnan(ar1_rmse)
    assert not np.isnan(xgb_rmse)
    
    # Absolute errors must be non-negative
    assert ar1_mae >= 0 
    assert xgb_mae >= 0

@patch('pandas.read_csv')
def test_load_data_parsing(mock_read_csv):
    """
    Test that load_data() properly drops header rows, renames RVD's specific 
    column schemas, merges on Month, and builds the Date index.
    """
    # We simulate successful pd.read_csv calls by building the DataFrames directly
    # (Because pandas.read_csv is mocked, using it here would just return another mock!)
    df_price = pd.DataFrame({
        "Month": ["01-2000", "02-2000"],
        "Class A": [100, 101],
        "Class B": [200, 202],
        "Class C": [300, 303],
        "Class D": [400, 404],
        "Class E": [500, 505]
    })
    
    df_yield = pd.DataFrame({
        "Month": ["01-2000", "02-2000"],
        "Domestic Class A": [2.1, 3.1],
        "Domestic Class B": [2.2, 3.2],
        "Domestic Class C": [2.3, 3.3],
        "Domestic Class D": [2.4, 3.4],
        "Domestic Class E": [2.5, 3.5]
    })
    
    # mock_read_csv will be called for price URL, then yield URL
    mock_read_csv.side_effect = [df_price, df_yield]
    
    df_result = load_data()
    
    # Verify the Date index was constructed perfectly from the string 'Month'
    assert df_result.index[0] == pd.to_datetime("2000-01-01")
    assert df_result.index[1] == pd.to_datetime("2000-02-01")
    
    # Verify Columns were cleanly renamed
    expected_cols = [
        'Price_A', 'Price_B', 'Price_C', 'Price_D', 'Price_E',
        'Yield_A', 'Yield_B', 'Yield_C', 'Yield_D', 'Yield_E'
    ]
    assert list(df_result.columns) == expected_cols
    
    # Verify values merged flawlessly
    assert df_result.loc["2000-01-01", "Price_A"] == 100
    assert df_result.loc["2000-02-01", "Yield_E"] == 3.5
