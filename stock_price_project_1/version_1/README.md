# Stock Price Prediction: S&P 500 (Version 1)
**Warning:** This version uses a single variable linear regression (days since 2004 → price).
It absolutely sucks at capturing real market behavior!

## Overview
This is **Version 1** of the **stock_price_project_1**, stored in the version_1/ directory. It demonstrates a very simple approach for predicting the S&P 500 closing price based solely on the passage of time (i.e., “days since 2004-01-01”).

**Result:** Because a single feature cannot capture market volatility or trends, the predictions are almost always unrealistic. This version is primarily educational to illustrate the basics of:

- Pulling data from Yahoo Finance
- Splitting into train/test
- Implementing a naive linear regression from scratch

## Directory Structure
version_1/
├── src/
│   ├── __init__.py
│   ├── data_collection.py
│   ├── model_training.py
│   └── main.py
├── README.md
└── requirements.txt

**data_collection.py**
Fetches historical S&P 500 daily closing data via yfinance.
Converts the date index to a numeric feature (“days_since_2004”), then splits into train/test.

**model_training.py**
Implements a single linear regression using gradient descent.

feature_scaling() (optional, though minimal with just one feature)
cost_function(), gradient_calculation(), gradient_descent()
Basic plotting of the cost over iterations

**main.py**

Loads and scales data
Trains the single-feature model
Plots the training cost
Provides a simple user interface (ask_for_input()) to predict future prices by specifying a date

**requirements.txt**
Basic Python dependencies (e.g., numpy, pandas, matplotlib, yfinance).

## How to Use
**Install Dependencies**

pip install -r requirements.txt
Run

python src/main.py

**The script will:**
Pull S&P 500 data from Yahoo Finance.
Train a single-variable regression on “days_since_2004.”
Plot the cost vs. iteration.
Ask for a future date to predict (spoiler: the prediction is almost never close to reality).

## Why This Model “Sucks”
**Single Feature:**
A single numeric feature (time) cannot possibly capture real-world stock fluctuations.
**Linear Assumption:**
Even if we tried polynomial transformations of time, pure “time-based” predictions for stock prices are inadequate.
**Ignoring Market Complexity:**
No market indicators, no volume, no returns, no macro data. This model has no context for actual market drivers.
In short, this version is a stepping stone—helpful for seeing how gradient descent and basic regression code work, but terrible for forecasting the real S&P 500.
