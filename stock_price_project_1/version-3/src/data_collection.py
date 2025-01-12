import yfinance as yf
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

# Creating the ticker object for S&P 500 and getting daily data for past 20 years
sp500_ticker = yf.Ticker("^GSPC")
vix_ticker = yf.Ticker("^VIX")

sp500_data = sp500_ticker.history(interval='1d', start='2004-01-01', end='2024-01-01')
vix_data = vix_ticker.history(interval='1d', start='2004-01-01', end='2024-01-01')

sp500_data['days_since_2004'] = (sp500_data.index - pd.to_datetime("2004-01-01", utc=True)).days  # B/c I'm using a regression model, I have to convert the date into numerical days since 2004, while accounting for holidays/weekends.
sp500_data.reset_index(inplace=True)
sp500_data['daily_return'] = sp500_data['Close'].pct_change(1)
sp500_data['ma_5'] = sp500_data['Close'].rolling(window=5).mean()
sp500_data['ma_20'] = sp500_data['Close'].rolling(window=20).mean()
sp500_data['volatility_5'] = sp500_data['Close'].rolling(window=5).std()
sp500_data['avg_gain'] = sp500_data['Close'] - sp500_data['Close'].shift(1)
sp500_data['avg_gain'] = sp500_data['avg_gain'].where(sp500_data['avg_gain'] > 0, 0)
sp500_data['avg_gain'] = sp500_data['avg_gain'].rolling(window=14).mean()
sp500_data['avg_loss'] = sp500_data['Close'] - sp500_data['Close'].shift(1)
sp500_data['avg_loss'] = sp500_data['avg_loss'].where(sp500_data['avg_loss'] < 0, 0)
sp500_data['avg_loss'] = sp500_data['avg_loss'].abs()
sp500_data['avg_loss'] = sp500_data['avg_loss'].rolling(window=14).mean()
sp500_data['rsi'] = 100 - (100)/(1 + (sp500_data['avg_gain'])/(sp500_data['avg_loss']))
sp500_data['day_of_week'] = sp500_data['Date'].dt.dayofweek
sp500_data = sp500_data.merge(vix_data[['Close', 'Open', 'High', 'Low', 'Volume']], how='left', left_on='Date', right_on='Date', suffixes=['_sp500', '_vix'])
sp500_data['target_tomorrow_open'] = sp500_data['Open_sp500'].shift(-1)
sp500_data = pd.get_dummies(sp500_data, columns=['day_of_week'], drop_first=True)
bool_cols = ['day_of_week_1', 'day_of_week_2', 'day_of_week_3', 'day_of_week_4']
for col in bool_cols:
    sp500_data[col] = sp500_data[col].astype(int)
sp500_data = sp500_data.iloc[20:]  # skip earliest 20 rows




sp500_data.drop('Date', axis=1, inplace=True)
sp500_data.drop('Dividends', axis=1, inplace=True)
sp500_data.drop('Stock Splits', axis=1, inplace=True)

column_names = [
    'Open_sp500',
    'High_sp500',
    'Low_sp500',
    'Close_sp500',
    'Volume_sp500',
    'days_since_2004',
    'daily_return',
    'ma_5',
    'ma_20',
    'volatility_5',
    'avg_gain',
    'avg_loss',
    'rsi',
    'day_of_week_1',
    'day_of_week_2',
    'day_of_week_3',
    'day_of_week_4'
]


X_data = sp500_data[column_names].to_numpy()
y_data = sp500_data['target_tomorrow_open'].to_numpy()
y_data_original = sp500_data['target_tomorrow_open'].to_numpy()


train_size = int(len(X_data) * 0.9)
X_train_no_fs, X_test = X_data[:train_size], X_data[train_size:]
y_train, y_test = y_data[:train_size], y_data[train_size:]
y_train_original, y_test_original = y_data_original[:train_size], y_data_original[train_size:]

print("Check for zero std in X_train_no_fs:")
stds = np.std(X_train_no_fs, axis=0)
print("stds:", stds)
print("Any zero or NaN in stds?", (stds == 0) | np.isnan(stds))



mu = np.mean(X_train_no_fs,axis=0)
sigma = np.std(X_train_no_fs,axis=0)
X_train = (X_train_no_fs - mu)/sigma
X_test = (X_test - mu)/sigma

y_len = y_train.shape[0]
y_test_len = y_test.shape[0]

for i in range(y_len - 1):
    if y_train_original[i+1] > y_train_original[i]:
        y_train[i] = 1
    else:
        y_train[i] = 0

y_train[4510] = 0

print(y_train)

for i in range(y_test_len - 1):
    if y_test_original[i+1] > y_test_original[i]:
        y_test[i] = 1
    else:
        y_test[i] = 0

y_test[501] = 0

print(y_test)
