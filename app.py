
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import pickle
import os

# Page config
st.set_page_config(
    page_title="Energy Demand Forecasting",
    page_icon="⚡",
    layout="wide"
)

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv('models/df_features.csv', index_col='Datetime', parse_dates=True)
    X_test = pd.read_csv('models/X_test.csv', index_col='Datetime', parse_dates=True)
    y_test = pd.read_csv('models/y_test.csv', index_col='Datetime', parse_dates=True)
    return df, X_test, y_test

@st.cache_resource
def load_models():
    xgb_final = XGBRegressor()
    xgb_final.load_model('models/xgb_final.json')

    q10 = XGBRegressor()
    q10.load_model('models/xgb_quantile_0.1.json')

    q50 = XGBRegressor()
    q50.load_model('models/xgb_quantile_0.5.json')

    q90 = XGBRegressor()
    q90.load_model('models/xgb_quantile_0.9.json')

    with open('models/results.pkl', 'rb') as f:
        results = pickle.load(f)

    return xgb_final, q10, q50, q90, results

df, X_test, y_test = load_data()
xgb_final, q10, q50, q90, results = load_models()

# Sidebar
st.sidebar.title("Energy Demand Forecasting")
st.sidebar.markdown("""
This app demonstrates a three-model energy demand forecasting system built on
16.5 years of hourly consumption data from the PJM East region of the US grid.

**Models compared:**
- SARIMA — statistical baseline
- XGBoost — gradient boosting with engineered features
- LSTM — deep learning sequential model

**Best model:** XGBoost with 0.78% MAPE
""")

horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    options=[24, 48, 168],
    format_func=lambda x: f"{x} hours ({x//24} day{'s' if x > 24 else ''})"
)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with XGBoost, PyTorch, Statsmodels, and Streamlit")

# Main title
st.title("Energy Demand Forecasting System")
st.markdown("PJM East Region — Hourly Consumption in Megawatts (MW)")
st.markdown("---")

# Generate forecasts
feature_cols = [col for col in X_test.columns]
y_test_values = y_test.values.flatten()

pred_median = q50.predict(X_test)
pred_lower  = q10.predict(X_test)
pred_upper  = q90.predict(X_test)
pred_point  = xgb_final.predict(X_test)

# Forecast chart
st.subheader("Forecast with Uncertainty Band")

n_history = 168
n_forecast = horizon

history_index  = X_test.index[:n_history]
forecast_index = X_test.index[n_history:n_history + n_forecast]

history_actual   = y_test_values[:n_history]
forecast_actual  = y_test_values[n_history:n_history + n_forecast]
forecast_median  = pred_median[n_history:n_history + n_forecast]
forecast_lower   = pred_lower[n_history:n_history + n_forecast]
forecast_upper   = pred_upper[n_history:n_history + n_forecast]

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(history_index, history_actual,
        color='steelblue', linewidth=1.0,
        label='Historical Demand')

ax.plot(forecast_index, forecast_actual,
        color='steelblue', linewidth=1.0,
        linestyle=':', alpha=0.5,
        label='Actual (forecast period)')

ax.plot(forecast_index, forecast_median,
        color='orange', linewidth=1.2,
        linestyle='--', label='XGBoost Forecast')

ax.fill_between(forecast_index,
                forecast_lower, forecast_upper,
                alpha=0.3, color='orange',
                label='80% Prediction Interval')

ax.axvline(x=forecast_index[0], color='gray',
           linestyle='--', linewidth=0.8, alpha=0.7)

ax.set_xlabel('Date')
ax.set_ylabel('Megawatts (MW)')
ax.legend(loc='upper right')
plt.tight_layout()
st.pyplot(fig)

# Metrics row
st.markdown("---")
st.subheader("XGBoost Point Forecast Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("MAE",  "245.76 MW")
col2.metric("RMSE", "335.26 MW")
col3.metric("MAPE", "0.78%")

# Model comparison table
st.markdown("---")
st.subheader("Model Comparison")

results_df = pd.DataFrame(results)
results_df = results_df.set_index('model')
results_df.columns = ['MAE (MW)', 'RMSE (MW)', 'MAPE (%)']
results_df = results_df.round(2)

st.dataframe(results_df, use_container_width=True)

st.markdown("---")
st.subheader("About This Project")
st.markdown("""
This forecasting system was built as part of a data science portfolio project
demonstrating end-to-end machine learning engineering across three model families.

**Key findings:**
- XGBoost with engineered temporal features achieves 0.78% MAPE — production grade accuracy
- lag_1 accounts for 65.4% of XGBoost predictive power, validating EDA findings
- LSTM outperforms SARIMA but underperforms XGBoost on this structured tabular dataset
- Quantile regression provides calibrated uncertainty intervals for capacity planning

**Further work:** Conformal prediction, N-BEATS architecture, ensemble modelling,
incorporation of weather features
""")
