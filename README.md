# Energy Demand Forecasting System

A production-grade time series forecasting pipeline built on 16.5 years of hourly
energy consumption data from the PJM East region of the US electrical grid. The
system compares three model families of increasing sophistication — a statistical
baseline, a gradient boosting model, and a deep learning sequential model — and
deploys the best performer with calibrated uncertainty quantification through an
interactive Streamlit application.

---

## Results Summary

| Model | MAE (MW) | RMSE (MW) | MAPE |
|-------|----------|-----------|------|
| SARIMA | 2,872.00 | 3,905.03 | 8.38% |
| XGBoost | 245.76 | 335.26 | **0.78%** |
| LSTM | 444.41 | 554.86 | 1.48% |

XGBoost with engineered temporal features achieves **0.78% MAPE** — well within
the sub-3% threshold used by professional grid operators — representing a 91.4%
improvement over the SARIMA statistical baseline.

---

## Project Structure

```
energy-forecasting/
│
├── data/
│   └── raw/                        # PJME_hourly.csv — source data
│
├── models/
│   ├── xgb_final.json              # Trained XGBoost model
│   ├── xgb_quantile_0.1.json       # Q10 quantile model
│   ├── xgb_quantile_0.5.json       # Q50 quantile model
│   ├── xgb_quantile_0.9.json       # Q90 quantile model
│   ├── lstm_model.pth              # Trained LSTM weights
│   ├── sarima_model.pkl            # Fitted SARIMA model
│   ├── df_features.csv             # Engineered feature dataset
│   ├── X_train.csv / X_test.csv    # Train/test feature splits
│   ├── y_train.csv / y_test.csv    # Train/test target splits
│   └── results.pkl                 # Model evaluation results
│
├── app.py                          # Streamlit application
├── notebook.ipynb                  # End-to-end development notebook
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** [PJM Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
— Kaggle

**Region:** PJM East (PJME) — covers the Mid-Atlantic United States including
parts of New Jersey, Delaware, Maryland, Virginia, and Washington D.C.

**Coverage:** January 2002 to August 2018 — 145,362 hourly observations after
removing 4 duplicate timestamps caused by daylight saving time transitions.

**Target variable:** PJME_MW — estimated energy consumption in Megawatts.

**Key statistics:**

| Metric | Value |
|--------|-------|
| Mean demand | 32,080 MW |
| Minimum demand | 14,544 MW |
| Maximum demand | 62,009 MW |
| Standard deviation | 6,464 MW |

---

## Methodology

### Phase 1 — Exploratory Data Analysis

Visual and statistical analysis revealed a three-level seasonal structure driving
demand patterns:

**Annual cycle** — bimodal peaks in winter (January to March) driven by heating
load and summer (July to September) driven by air conditioning load. The summer
peak consistently exceeds the winter peak, with August recording the annual maximum.
Spring and autumn shoulder seasons produce demand troughs as mild temperatures
reduce both heating and cooling requirements.

**Weekly cycle** — weekday consumption is consistently higher than weekend
consumption, reflecting reduced commercial and industrial activity on Saturdays
and Sundays.

**Daily cycle** — a dual-peak pattern within each 24-hour period, with a morning
rise as commercial activity begins and an evening peak around 18:00 as residential
and commercial demand converge.

The Augmented Dickey-Fuller test returned an ADF statistic of -19.89 against a
1% critical value of -3.43 with a p-value of 0.00, confirming stationarity at all
significance levels. No differencing was required prior to modelling.

ACF and PACF analysis confirmed strong short-term autocorrelation with a dominant
24-hour seasonal cycle, directly informing both the SARIMA configuration and the
feature engineering strategy.

### Phase 2 — Feature Engineering

22 features were engineered from the raw timestamp and target variable across four
groups. All features use shift(1) before any window calculation to prevent data
leakage.

**Calendar features** encode hour, day of week, month, quarter, year, day of year,
and a binary weekend flag derived from the weekly cycle identified in EDA.

**Lag features** encode past values at 1, 24, 48, and 168 hour intervals,
directly informed by significant ACF and PACF spikes. Lag 24 and lag 168 capture
the daily and weekly repeat patterns respectively.

**Rolling window features** encode the 24-hour and 168-hour rolling mean and
standard deviation, providing models with context about recent demand levels and
volatility rather than isolated past points.

**Fourier terms** encode the 24-hour and weekly cycles as sine and cosine pairs,
resolving the cyclical boundary problem that raw integer time features cannot
handle. Two orders per period capture both the broad seasonal swing and finer
intra-cycle structure including the dual morning and evening demand peaks.

### Phase 3 — Modelling

The dataset was split chronologically at January 1st 2017. All models trained on
January 2002 to December 2016 and evaluated on January 2017 to August 2018 —
preserving the temporal order to prevent data leakage.

**Layer 1 — SARIMA(2,0,1)(1,1,1,24)**

Parameters set directly from ADF and ACF/PACF findings rather than exhaustive
search. All six parameters returned p-values of 0.000 confirming statistical
significance. The Ljung-Box test confirmed residuals approximate white noise,
meaning the model captured the core temporal structure. Heavy-tailed residuals
(kurtosis 5.43) reflect the model's difficulty with extreme demand events.

SARIMA successfully replicates the timing and rhythm of daily cycles but
systematically underestimates demand peaks, remaining anchored to a narrow band
while actual consumption reaches 40,000 to 45,000 MW during cold weather events.
This structural limitation — the model has access only to the recent series with
no external contextual information — motivates the transition to feature-rich
models.

**Layer 2 — XGBoost with Engineered Features**

Trained on 131,299 observations with early stopping monitored over 100
consecutive trees. The optimal model was found at iteration 2,996 with a final
test RMSE of 335 MW.

Feature importance analysis produced a clear hierarchy directly validating the
EDA findings:

| Feature | Importance |
|---------|------------|
| lag_1 | 65.4% |
| lag_24 | 7.3% |
| lag_168 | 6.8% |
| sin_168_1 | 6.3% |
| cos_24_1 | 3.9% |

lag_1 alone drives 65.4% of predictive power, confirming that the dominant PACF
spike at lag 1 identified during EDA represented the strongest predictive signal
in the data. The engineered features gave XGBoost the contextual precision to
track both demand peaks and troughs accurately across the full 19-month test
period, achieving a MAPE of 0.78%.

**Layer 3 — LSTM Neural Network**

A two-layer stacked LSTM with hidden size 128 and dropout 0.2, trained on sliding
windows of 168 hours (one full week) to predict the subsequent hour's demand.
207,489 trainable parameters. Trained for 20 epochs on a T4 GPU with gradient
clipping (max norm 1.0) and a ReduceLROnPlateau scheduler.

Training loss declined consistently from 0.016 at epoch 1 to 0.00046 at epoch 20
with no overfitting signal — validation loss tracked training loss throughout,
confirming the regularisation strategy was effective.

Visual analysis revealed a characteristic asymmetry — the LSTM tracks demand peaks
confidently across both winter and summer extremes but slightly overshoots overnight
troughs, reflecting the sequential memory weighting sustained patterns more heavily
than sharp momentary dips. The LSTM underperforms XGBoost on this dataset because
energy demand follows strong regular patterns that explicit engineered features
capture more precisely than learned sequential representations. On datasets with
irregular or difficult-to-encode temporal patterns the LSTM architecture would be
expected to outperform feature-based approaches.

The two models are characteristically complementary — XGBoost provides precise
trough estimation while the LSTM reinforces peak confidence — suggesting an
ensemble as a natural extension.

### Phase 4 — Uncertainty Quantification

Quantile regression was implemented for XGBoost by training separate models for
the 10th, 50th, and 90th percentiles, producing an 80% prediction interval around
each forecast.

| Metric | Value |
|--------|-------|
| Target coverage | 80.00% |
| Achieved coverage | 67.79% |
| Average interval width | 687.78 MW |
| Width as % of mean demand | 2.21% |

The intervals are slightly narrow — 67.79% coverage against the 80% target —
reflecting a known characteristic of quantile regression on high-accuracy tree
models. XGBoost's median forecast MAE of 246 MW means the quantile models learned
that true values almost always sit close to the median, placing boundaries
accordingly. Undercoverage arises from the small proportion of extreme weather-driven
demand events that exceed the narrow band. The average interval width of 687 MW
(2.21% of mean demand) is operationally meaningful for capacity planning.

---

## Key Findings

XGBoost with domain-informed feature engineering outperforms a deep learning
approach on this structured tabular time series, demonstrating that architectural
complexity does not guarantee superior performance when the problem structure is
well understood. The 91.4% improvement over SARIMA is driven primarily by lag_1
— a single engineered feature that gives the model direct access to the most
recent observation, something a pure statistical model trained on a short window
cannot replicate.

The progression from SARIMA to XGBoost to LSTM illustrates a central principle
in applied forecasting: understand the data structure first, engineer features
that encode it explicitly, then evaluate whether additional model complexity
provides genuine improvement.

---

## Setup and Usage

### Requirements

```
pandas
numpy
matplotlib
seaborn
statsmodels
pmdarima
scikit-learn
xgboost
torch
streamlit
pyngrok
pickle
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

### Running the Notebook

Open `notebook.ipynb` in Google Colab or a local Jupyter environment. The notebook
is structured sequentially — run cells from top to bottom. A GPU runtime is
recommended for the LSTM training phase.

Download `PJME_hourly.csv` from the Kaggle dataset linked above and upload it to
your runtime before running Phase 1.

### Running the App

```bash
streamlit run app.py
```

On Google Colab use ngrok to expose the local server:

```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
public_url = ngrok.connect(8501)
print(public_url)
```

---

## Further Work

**Live data integration** — PJM publishes real-time grid consumption data via a
public API. Connecting the feature pipeline to this feed would enable genuine
rolling forecasts with always-current lag values, transforming the system from a
historical demonstration to a live production forecasting tool. This requires a
persistent server and a scheduled retraining pipeline to account for gradual
concept drift.

**Recursive multi-step forecasting** — implementing 24 to 48 hour ahead forecasts
using the model's own predictions as lag inputs for subsequent steps, with
explicitly widening prediction intervals to communicate compounding uncertainty
honestly.

**Conformal prediction** — a post-processing technique that adjusts interval width
using a calibration set to guarantee exact coverage at any target level regardless
of the underlying model, addressing the current undercoverage in the tails.

**Ensemble modelling** — combining XGBoost and LSTM predictions to leverage
XGBoost's precise trough estimation and the LSTM's peak confidence, potentially
outperforming either model individually.

**Neural architecture extension** — N-BEATS or Temporal Fusion Transformer as
more powerful alternatives to the LSTM, with built-in interpretability through
attention mechanisms providing timestep-level feature importance.

**Weather feature integration** — incorporating temperature and weather forecast
data as external features to address the primary remaining error source: demand
spikes and troughs driven by extreme weather events that the current models
must infer from historical patterns alone.

---

## Tech Stack

Python, Pandas, NumPy, Statsmodels, pmdarima, Scikit-learn, XGBoost, PyTorch,
Matplotlib, Seaborn, Streamlit, Google Colab, ngrok

---

## Data Source

Godahewa, R., Bergmeir, C., Webb, G., Hyndman, R., & Montero-Manso, P. (2021).
PJM Hourly Energy Consumption. Kaggle.
https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
