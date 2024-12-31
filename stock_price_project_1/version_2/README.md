## Stock Price Prediction: S&P 500 (Version 2)
**Note:** This version leverages **multiple features** (technical indicators, rolling averages, day-of-week, VIX data, etc.) in a linear regression, surpassing the simplistic approach of Version 1.

# Overview
This is **Version 2** of the **stock_price_project_1**, stored in the **version_2/** directory. It demonstrates a **multi-feature linear regression** approach to predict S&P 500 prices, going beyond the single “days since 2004” feature used in Version 1.

**Result:** By incorporating more features (rolling averages, RSI, day-of-week, and optional VIX data), the model often yields more realistic forecasts than a naive time-based approach. It still relies on a linear model, but the additional indicators help capture market trends more effectively.

# Directory Structure
```version_2/
├── src/ │
  ├── init.py
│ ├── data_collection.py
│ ├── model_training.py
│ └── main.py
├── README.md
└── requirements.txt
```
**data_collection.py**

+ Fetches historical S&P 500 daily data from Yahoo Finance (and optionally VIX).
+ Creates multiple features (rolling averages, volatility, RSI, day-of-week, etc.).
+ Splits data into train/test sets chronologically.
+ Scales features (and target, if desired).

**model_training.py**

+ Implements multi-feature linear regression via gradient descent:
+ cost_function(), gradient_calculation(), gradient_descent()
+ Provides cost-plotting functionality over iterations.

**main.py**

+ Loads and scales data from data_collection.py
+ Trains the multi-feature model (weights = one per indicator)
+ Plots cost vs. iteration and/or predicted vs. actual to gauge performance.

**requirements.txt**
+ Basic Python dependencies (numpy, pandas, matplotlib, yfinance, etc.) for multi-feature regression.

# How to Use
**Install Dependencies**
```bash
pip install -r requirements.txt
```

**Run**
```bash
python src/main.py
```

**The script will:**

1. Fetch S&P 500 and optional VIX data.
2. Compute rolling averages, RSI, and day-of-week features.
3. Merge, scale, and split the dataset into training (e.g., 90%) and testing.
4. Train a multi-feature linear regression with gradient descent.
5. Plot the training cost (and possibly a predicted vs. actual scatter plot).

# Why This Model is Better Than Version 1

**Multiple Features:**
Moving averages (e.g., 5-day, 20-day), RSI, volatility, day-of-week, and optional VIX data each provide signals far more relevant than mere time elapsed.

**Captures Some Market Complexity:**
Though still a linear approach, additional features let the model track short-term trends, momentum, and volatility more closely than a single “days since 2004” metric.

**Greater Realism:**
Compared to Version 1, you’ll see forecasts that align more closely with actual price movements—though there’s still room for advanced models (ARIMA, LSTM, Transformers) or more fundamental/macro data.

In short, **Version 2** is a substantial improvement over Version 1’s one-dimensional view, offering a practical next step toward a more robust stock forecasting model.
