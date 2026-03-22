# Hybrid Temporal Forecaster — BTC Historical Data

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Analysis Pipeline](#analysis-pipeline)
  - [1. Data Ingestion & Preprocessing](#1-data-ingestion--preprocessing)
  - [2. Stationarity Analysis](#2-stationarity-analysis)
  - [3. Seasonal Decomposition](#3-seasonal-decomposition)
  - [4. ACF & PACF Analysis](#4-acf--pacf-analysis)
  - [5. ARIMA Model Training](#5-arima-model-training)
  - [6. Model Evaluation](#6-model-evaluation)
- [Project Structure](#project-structure)
- [Installation & Requirements](#installation--requirements)
- [Getting Started](#getting-started)
- [Results Summary](#results-summary)
- [Technologies Used](#technologies-used)

---

## Overview
This project provides a comprehensive framework for analyzing and forecasting **Bitcoin (BTC/USD)** historical price data using classical time-series modeling techniques. The core focus is on the **ARIMA (AutoRegressive Integrated Moving Average)** model, and the analysis rigorously investigates the impact of **data stationarity** on forecasting accuracy.

The entire pipeline—from raw data ingestion through to model evaluation and visualization—is implemented in a single, well-documented Jupyter Notebook. The analysis demonstrates a key principle of time-series forecasting: that **transforming non-stationary data into a stationary form (via log returns) dramatically improves ARIMA model performance**.

---

## Key Features
- **Massive Dataset Handling**: Processes over **7.4 million** rows of minute-resolution BTC/USD OHLCV data, spanning from **Jan 2012 to Mar 2026**.
- **Smart Data Preprocessing**: Converts Unix timestamps, resamples data from 1-minute to 1-hour intervals, filters to the 2020–present window (~54,000 hourly observations), and handles missing values.
- **Rigorous Stationarity Testing**: Implements both the **Augmented Dickey-Fuller (ADF)** and **Kwiatkowski-Phillips-Schmidt-Shin (KPSS)** statistical tests to formally determine the stationarity of the time series.
- **Log-Return Transformation**: Converts raw close prices into log returns (`log(close_t / close_{t-1})`) to stabilize variance and achieve stationarity—a critical prerequisite for ARIMA modeling.
- **Seasonal Decomposition**: Performs additive seasonal decomposition with a 24-hour (daily) period to investigate trend, seasonal, and residual components.
- **ACF/PACF Analysis**: Uses Autocorrelation and Partial Autocorrelation function plots to determine appropriate ARIMA model orders.
- **Automated Model Selection**: Utilizes `pmdarima.auto_arima` to systematically search for and identify the optimal $(p, d, q)$ parameters for ARIMA models on both stationary and non-stationary data.
- **Head-to-Head Model Comparison**:
    - **Non-Stationary Model**: ARIMA fitted directly on raw close prices.
    - **Stationary Model**: ARIMA fitted on transformed stationary log returns.
- **Comprehensive Evaluation Metrics**: Assesses forecast quality using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.
- **Publication-Quality Visualizations**: Generates comparative forecast plots (actual vs. predicted), zoomed-in forecast horizons, decomposition charts, and correlation plots—all saved as high-resolution PNGs.
- **Model Persistence**: Trained ARIMA models are serialized using `joblib` for later reuse without retraining.

---

## Analysis Pipeline

### 1. Data Ingestion & Preprocessing
- **Source**: `btcusd_1-min_data.csv` — a CSV containing minute-level OHLCV (Open, High, Low, Close, Volume) data.
- **Raw shape**: ~7,476,160 rows × 6 columns.
- **Timestamp handling**: Unix epoch timestamps (in seconds) are parsed with `pd.to_datetime(unit='s')`.
- **Resampling**: Data is aggregated to **1-hour** intervals using proper OHLCV aggregation rules (first open, max high, min low, last close, sum volume).
- **Date Filtering**: Data is cropped to the **January 2020–present** window, yielding ~54,487 hourly observations after dropping NaN rows.

### 2. Stationarity Analysis
A custom `stationarity_test()` function runs two complementary statistical tests:

| Test | Null Hypothesis | Result on Raw Close | Result on Log Returns |
|------|-----------------|--------------------|-----------------------|
| **ADF (Augmented Dickey-Fuller)** | Series is non-stationary | ❌ Non-Stationary (p ≈ 0.60) | ✅ Stationary |
| **KPSS** | Series is stationary | ❌ Non-Stationary (p ≈ 0.01) | ✅ Stationary |

Both tests confirm that raw BTC close prices are **non-stationary**, while log returns are **stationary**—validating the transformation approach.

### 3. Seasonal Decomposition
- Uses `seasonal_decompose()` with an **additive** model and a **24-hour period**.
- Decomposes the time series into **Observed**, **Trend**, **Seasonal**, and **Residual** components.
- Result: No significant daily seasonal pattern is found in the raw close price data.

### 4. ACF & PACF Analysis
- **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** plots are generated for both the raw close prices and the stationary log returns.
- These plots guide the selection of ARIMA model orders $(p, d, q)$.

### 5. ARIMA Model Training
Using `pmdarima.auto_arima`, the analysis trains two separate ARIMA models:

1. **Stationary ARIMA**: Trained on log return data. `auto_arima` searches for optimal $(p, d, q)$ parameters.
2. **Non-Stationary ARIMA**: Trained directly on raw close prices for comparison.

Both models are fitted using `statsmodels.tsa.arima.model.ARIMA` and serialized to disk via `joblib`.

### 6. Model Evaluation
Models are evaluated on a held-out test set using:
- **MAE (Mean Absolute Error)**: Measures average absolute forecast error.
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily.

The **stationary (log-return) ARIMA model** consistently outperforms the non-stationary model, confirming the importance of data preprocessing.

---

## Project Structure
```
Hybrid-Temporal-Forecaster---BTC-historical-data/
├── Hybrid Temporal Forecaster.ipynb   # Main analysis notebook
├── btcusd_1-min_data.csv              # Raw BTC/USD minute data (user-provided)
├── figures/                           # Generated plots (auto-created)
│   ├── 01_log_returns.png
│   ├── 02_btc_raw_price.png
│   ├── 03_seasonal_decomp_daily.png
│   ├── 04_acf_pacf_stationary.png
│   ├── 05_acf_pacf_nonstationary.png
│   ├── 06_arima_nonstationary_forecast.png
│   └── 07_arima_stationary_forecast.png
├── models/                            # Serialized models (auto-created)
│   ├── arima_stationary.joblib
│   └── arima_nonstationary.joblib
└── README.md
```

## Installation & Requirements
Ensure you have **Python 3.8+** installed. Then install the required libraries:
```bash
pip install pandas numpy matplotlib statsmodels pmdarima scikit-learn scipy joblib
```
Or, if you prefer conda:
```bash
conda install pandas numpy matplotlib statsmodels scikit-learn scipy joblib
pip install pmdarima
```

## Getting Started
1. **Clone** this repository.
2. **Place** your historical data file (e.g., `btcusd_1-min_data.csv`) in the project root directory.
3. **Launch** Jupyter:
   ```bash
   jupyter notebook
   ```
4. **Open** `Hybrid Temporal Forecaster.ipynb` and run the cells sequentially.

> **Note**: Processing 7.4M+ rows may require significant memory. A machine with at least **8 GB RAM** is recommended.

## Results Summary
The analysis confirms that:
- Raw BTC close prices are **non-stationary** (failing both ADF and KPSS tests).
- **Log returns** provide a stable, stationary basis for time-series forecasting.
- The **stationary ARIMA model** significantly outperforms the raw-price ARIMA model in terms of both MAE and RMSE.
- ARIMA models exhibit **mean-reverting behavior** over longer forecast horizons, a fundamental characteristic of stationary time-series models.

## Reproducibility Guide

To fully reproduce the results of this analysis, follow these steps carefully:

### Step 1: Environment Setup
Create an isolated environment to avoid dependency conflicts:
```bash
# Using venv
python -m venv hybrid-forecaster-env
source hybrid-forecaster-env/bin/activate   # Linux/Mac
# hybrid-forecaster-env\Scripts\activate    # Windows

# Install all dependencies
pip install pandas numpy matplotlib statsmodels pmdarima scikit-learn scipy joblib jupyter
```

### Step 2: Obtain the Dataset
- The analysis uses the **`btcusd_1-min_data.csv`** file containing minute-resolution BTC/USD OHLCV data.
- The CSV must have the following columns (with a header row): `timestamp`, `open`, `high`, `low`, `close`, `volume`.
- The `timestamp` column should be in **Unix epoch format (seconds)**.
- Place this file in the same directory as the notebook.

### Step 3: Run the Notebook
```bash
jupyter notebook "Hybrid Temporal Forecaster.ipynb"
```
- **Run all cells sequentially** (Kernel → Restart & Run All) to ensure reproducibility.
- Do **not** skip cells or run them out of order, as later cells depend on variables and DataFrames created in earlier ones.

### Step 4: Expected Outputs
After a successful run, you should see:
| Output | Location | Description |
|--------|----------|-------------|
| `01_log_returns.png` | `figures/` | Plot of hourly log returns |
| `02_btc_raw_price.png` | `figures/` | Plot of raw BTC close prices |
| `03_seasonal_decomp_daily.png` | `figures/` | Additive seasonal decomposition (24h period) |
| `04_acf_pacf_stationary.png` | `figures/` | ACF & PACF of stationary log returns |
| `05_acf_pacf_nonstationary.png` | `figures/` | ACF & PACF of non-stationary raw prices |
| `06_arima_nonstationary_forecast.png` | `figures/` | Forecast vs. actual for non-stationary ARIMA |
| `07_arima_stationary_forecast.png` | `figures/` | Forecast vs. actual for stationary ARIMA |
| `arima_stationary.joblib` | `models/` | Serialized stationary ARIMA model |
| `arima_nonstationary.joblib` | `models/` | Serialized non-stationary ARIMA model |

### Step 5: Validate Results
To confirm reproducibility, check the following:
1. **Stationarity Tests**: Raw close prices should fail both ADF (p > 0.05) and KPSS (p < 0.05) tests. Log returns should pass both.
2. **Model Performance**: The stationary ARIMA model should yield **lower MAE and RMSE** than the non-stationary model.
3. **Figures**: All 7 plots should be generated in the `figures/` directory.
4. **Models**: Both `.joblib` files should be present in the `models/` directory.

### Troubleshooting
| Issue | Solution |
|-------|----------|
| `MemoryError` during data loading | Use a machine with ≥ 8 GB RAM, or subsample the data |
| `auto_arima` runs very slowly | This is expected for large datasets; may take 10–30+ minutes |
| `InterpolationWarning` from KPSS test | This is normal—it means the test statistic is outside the lookup table range |
| Missing `figures/` or `models/` dirs | The notebook creates them automatically via `os.makedirs(..., exist_ok=True)` |
| Different ARIMA orders | Results may vary slightly depending on the dataset version and `pmdarima` version |

---

## Technologies Used
| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, resampling |
| `numpy` | Numerical computations, log transforms |
| `matplotlib` | Visualization and plotting |
| `statsmodels` | ADF/KPSS tests, ARIMA modeling, ACF/PACF plots, seasonal decomposition, Ljung-Box test |
| `pmdarima` | Automated ARIMA parameter selection (`auto_arima`) |
| `scikit-learn` | Evaluation metrics (MAE, RMSE) |
| `scipy` | Statistical utilities |
| `joblib` | Model serialization/persistence |

---
*Disclaimer: This project is for educational and research purposes only. Cryptocurrency markets are highly volatile and unpredictable. Past performance does not guarantee future results. Financial forecasting involves significant risk—use at your own discretion.*
