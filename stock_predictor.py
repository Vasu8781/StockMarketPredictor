# Import required libraries
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Download stock data for a company (e.g., Apple - 'AAPL')
stock_symbol = 'AAPL'
data = yf.download(stock_symbol, start='2020-01-01', end='2024-01-01')

# Show first few rows
print(data.head())
# Prepare the data
data = data.reset_index()
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].map(pd.Timestamp.toordinal)

# Features (X) and Target (y)
X = data[['Date']]
y = data['Close']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict the next 30 days
import datetime

future_dates = [datetime.date(2024, 1, 2) + datetime.timedelta(days=i) for i in range(30)]
future_dates_ordinal = [[pd.Timestamp(date).toordinal()] for date in future_dates]

predictions = model.predict(future_dates_ordinal)

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(data['Date'], y, label='Historical Prices')
plt.plot(future_dates_ordinal, predictions, label='Predicted Prices', color='red')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f'{stock_symbol} Stock Price Prediction')
plt.legend()
plt.show()
