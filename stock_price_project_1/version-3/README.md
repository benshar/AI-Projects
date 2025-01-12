# Stock Price Prediction: S&P 500 (Version 3 - Logistic Regression)
In this **Version 3** of the stock price project, we switch from **multi-feature linear regression** (Version 2) to a **logistic regression** approach. Instead of predicting tomorrow’s opening price numerically, we classify whether tomorrow’s opening price will be **up** (1) or **down** (0) relative to today’s.

## Overview
- **Goal:** Predict a **binary label** (1 = tomorrow is higher, 0 = tomorrow is lower) for the S&P 500’s opening price.
- **Method:** Logistic Regression implemented from scratch:
  - Uses a **sigmoid** (logistic) function
  - Calculates a **binary cross-entropy** (logistic) cost
  - Performs **gradient descent** to optimize parameters
- **Data & Features:**
  - Same technical indicators as Version 2 (moving averages, RSI, day-of-week, optional VIX columns, etc.)
  - Brought into a numpy array, then scaled and split into train/test.
- **Result:** Observed that the model often “defaults” to predicting 1 (i.e., “up”) for many rows, yielding around 50% accuracy.

## Directory Structure
```bash
version_3/ 
├── src/ 
│ ├── __init__.py 
│ ├── data_collection.py # Gathers data, merges S&P and VIX, creates features, splits into train/test 
│ ├── model_training.py # Logistic regression code: cost function, gradient, gradient descent 
│ └── main.py # Ties everything together, trains model, checks predictions 
├── README.md # You're reading it now 
└── requirements.txt # Dependencies (numpy, pandas, matplotlib, yfinance, etc.)
```

**data_collection.py**
Fetches historical S&P 500 daily data from Yahoo Finance (and optionally VIX).
Creates multiple features (rolling averages, volatility, RSI, day-of-week, etc.).
Splits data into train/test sets chronologically.
Scales features (and target, if desired).

**odel_training.py**
Implements multi-feature logistic regression:
cost_function() uses binary cross-entropy
gradient_calculation() uses sigmoid & partial derivatives
gradient_descent() loop

**main.py**
Loads the processed features/labels from data_collection.py
Trains the multi-feature model (weights = one per indicator)
Plots cost vs. iteration and/or predicted vs. actual to gauge performance.

**requirements.txt**
Basic Python dependencies:
numpy, pandas, matplotlib, yfinance, etc.

## How to Use
**Install Dependencies**
```python
pip install -r requirements.txt
```

**Run**
```python
src/main.py
```

**The script will:**

1. Pull data (S&P 500 and possibly VIX) from Yahoo Finance.
2. Compute rolling averages, RSI, and other features.
3. Merge, scale, and split the dataset into training (e.g., 90%) and testing.
4. Train a **multi-feature** logistic regression with gradient descent.
5. Plot the training cost (and possibly a predicted vs. actual scatter plot).

## Why This Model Defaults to 1s
1. **Slight Bullish Bias in the Data**
- The S&P 500 historically goes “up” a bit more than “down,” so around 55–60% of days are up.
- The logistic regression may converge to a near-constant “predict up” solution if the features provide weak day-to-day signals.

2. **Limited Features**
- Indicators like RSI, moving averages, or day-of-week might not strongly differentiate daily direction.
- As a result, the model finds minimal advantage in deviating from a near-constant guess of “1.”

3. **Model & Data Realities**
- Intraday or day-to-day moves can be noisy and somewhat random, especially in a broad index like the S&P 500.
- If you want a stronger edge, advanced features (fundamentals, macro, order book data) or different models might be necessary.

## Suggestions for Improvement
1. **Double-Check Label Alignment**
- Ensure row i’s features correspond to the day you compare for up/down.
2. **Add More or Different Features**
- Lagged prices (Close[t-1], Volume[t-1])
- Macroeconomic or fundamental data
- Alternative advanced indicators
3. **Tune Learning Rate / Iterations**
- If gradient descent doesn’t converge well, consider lowering alpha or raising iteration count.
- Or try a different solver (like scikit-learn’s LogisticRegression).
4. **Consider a Different Approach**
- Random forests, gradient-boosted trees, or neural networks might learn more complex patterns.
- Or incorporate more domain-specific signals (sector correlation, sentiment, etc.).

## Conclusion
This **Version 3** highlights **logistic regression** for daily up/down classification. The code runs correctly, but the **50% accuracy** and near-constant “1” predictions suggest day-to-day S&P 500 direction is not easily captured by these simple features in a linear logistic model. This is **not necessarily a bug—**it’s a normal outcome for short-term index direction prediction using standard indicators.





