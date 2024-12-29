import yfinance as yf
import pandas as pd

# Creating the ticker object for S&P 500 and getting daily data for past 20 years
sp500_ticker = yf.Ticker("^GSPC")

sp500_data = sp500_ticker.history(interval='1d', start='2004-01-01', end='2024-01-01')

# print(sp500_data.head())   <<  I used this command to get a preview of the data

sp500_close = sp500_data[['Close']].copy()

sp500_close['days_since_2004'] = (sp500_close.index - pd.to_datetime("2004-01-01", utc=True)).days  # B/c I'm using a regression model, I have to convert the date into numerical days since 2004, while accounting for holidays/weekends.
sp500_close.reset_index(inplace=True)
sp500_close.drop('Date', axis=1, inplace=True)


# print(sp500_close.head())  <<  I used this command to get a preview of the new data

# Checking for any non-trading day values
# print(sp500_close.isnull().sum()) I used this command to see if there were any null values in the entire column. The respons was 0

x_values = sp500_close[['days_since_2004']]
y_values = sp500_close[['Close']]

split_index = int(len(x_values) * 0.9)

# Creating the training and testing sets - 90% for training, 10% for testing
X_train = x_values.iloc[:split_index]
y_train = y_values.iloc[:split_index]
X_test  = x_values.iloc[split_index:]
y_test  = y_values.iloc[split_index:]

# print(X_train.head())       # Looking at datasets to be sure they're ready
# print(y_train.head())
# print(X_test.head())
# print(y_test.head())
