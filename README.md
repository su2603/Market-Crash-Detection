# Market Crash Detection System

## Overview

This system is designed to detect and predict financial market crashes using both traditional statistical indicators and machine learning models. It includes a Streamlit-based interactive dashboard for real-time visualization and analysis.

---

## Modules

### 1. `marketCrashDetectionApp.py`

This is the **Streamlit frontend app** for market crash detection. It offers:

- Data upload and cleaning
- Indicator computation
- Crash detection
- Interactive charts and dashboards
- ML-based prediction and visualization

### 2. `marketCrashIdentification.py`

This is the **core backend engine** containing:

- Data ingestion & validation
- Indicator computation
- Feature engineering
- Crash labeling
- ML model training (Isolation Forest + Random Forest)
- Reporting, backtesting, and visualization

---

## Core Features

### Technical Indicators

Implemented in both modules:

- **Daily Returns**: `% change` in closing prices
- **Log Returns**
- **SMA & EMA**: Short and long-term moving averages
- **Volatility**: Rolling standard deviation and GARCH-style estimation
- **Drawdown**: Decline from historical peak
- **RSI, MACD, Bollinger Bands**: Momentum and volatility tools

### Crash Detection Logic

Crash labels are created using:

- Daily return < -0.5%
- Rolling volatility > 2%
- Drawdown < -10%
- Z-score of returns > 2.5
- Multi-day negative returns (optional)

These conditions are aggregated into a final `Crash_Label`.

---

## Machine Learning Models

### Isolation Forest

- Detects outliers/anomalies in engineered features
- Flags "Is_Anomaly" and provides "Anomaly_Score"

### Random Forest Classifier

- Trained on labeled crash data
- Predicts crash probabilities (`Crash_Probability`)
- Includes feature importance analysis

---

## Feature Engineering

- Lag features (`Return_Lag_x`)
- Rolling mean, std, skew, kurt of returns
- Momentum indicators
- Volatility ratios and breakouts
- Regime detection (trend, return regime, volatility regime)

---

## Visualizations

### Streamlit Interactive Charts:

- Price with crash points
- Volatility with thresholds
- Drawdown shaded area
- Probability of crash
- Feature importance (ML)
- Daily return histogram

### Matplotlib Dashboard (from backend):

- 12 subplots with all key indicators
- Correlation heatmap
- Risk metrics summary
- Optionally saved to file

---

## Model Persistence

The system supports:

- Saving trained models with `joblib`
- Reloading for future prediction or deployment

```python
detector.save_model("crash_detection_model.pkl")
detector.load_model("crash_detection_model.pkl")
```

---

## Backtesting Engine

Allows evaluation of strategy:

- Predict crash if probability > threshold (e.g., 0.7)
- Compare predictions with actual crashes
- Outputs: accuracy, precision, recall, F1, confusion matrix

---

## Input Format

```csv
Date,Close
2024-01-01,57321.1
2024-01-02,57012.8
...
```

- Only `Date` and `Close` are required.
- `Volume` is optional (auto-generated if absent).

---

## Sample Report Output

```json
{
  "data_summary": {
    "total_observations": 6835,
    "date_range": "1997-07-01 to 2025-05-30",
    "price_range": "4300.86 to 62341.2",
    "total_return": "1248.23%"
  },
  "crash_analysis": {
    "total_crashes": 123,
    "crash_rate": "1.80%",
    "max_drawdown": "-41.73%"
  },
  "risk_metrics": {
    "daily_volatility": "1.12%",
    "annualized_volatility": "17.74%",
    "sharpe_ratio": "1.23",
    "var_95": "-2.13%",
    "skewness": "-0.37",
    "kurtosis": "4.91"
  }
}
```

---

## 📦 Dependencies

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `scipy`
- `joblib`, `dataclasses`, `datetime`
- `streamlit`, `plotly` (frontend app)

---

## 🚀 Running the App

### Streamlit Dashboard:

```bash
streamlit run marketCrashDetectionApp.py
```

You can upload your CSV or use sample data.

---

## Usage Guidelines

### 🔹 When to Use

- Monitoring for **potential market crashes**
- Evaluating **historical crash patterns**
- Enhancing **risk management strategies**
- **Backtesting** crash signals for portfolio protection

### 🔹 Who Can Use This

- Financial analysts
- Risk managers
- Quantitative researchers
- Asset managers and hedge funds

---

## Sample Workflow

### Step-by-Step (Streamlit App)

1. Launch the app:

   ```bash
   streamlit run marketCrashDetectionApp.py
   ```

2. Upload a CSV file with `Date` and `Close` columns  
   _(or let it generate synthetic data)_

3. Configure detection parameters in the sidebar:

   - Return threshold
   - Volatility and drawdown thresholds
   - Moving average windows

4. Click **Run Analysis** to start detection

5. Explore:
   - Crash alerts
   - Volatility trends
   - Drawdown patterns
   - ML-based crash probabilities
   - Download enriched dataset

---

### Project Context & Recommendations

**Project Objectives**
-This system is designed to serve as an early warning tool for financial market instability. Its goal is to:
-Help portfolio managers anticipate downturns before they become severe
-Provide quantitative triggers to adjust risk exposure
-Support post-crash analysis to improve future resilience

**This project is best used in:**

-Financial research platforms for historical crash evaluation
-Risk dashboards that alert managers of stress conditions
-Quantitative trading systems to avoid exposure during turmoil
-Decision support systems in banks, hedge funds, and asset management firms

---
