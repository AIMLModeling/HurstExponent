import yfinance as yf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sample_size = 1000
scalar = 100

def get_hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""
    lags = range(2, max_lag)
    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]

np.random.seed(123)
df = pd.DataFrame(data={"mean_rev": np.log(np.random.randn(sample_size) + scalar),
                        "gbm": np.log(np.cumsum(np.random.randn(sample_size)) + scalar),
                        "trending": np.log(np.cumsum(np.random.randn(sample_size) + 1) + scalar)})

plt.plot(df['mean_rev'], label ='mean reversion')
plt.plot(df['gbm'], label ='geometric Brownian motion')
plt.plot(df['trending'], label ='Trending')
plt.title('Generated Time Series')
plt.legend()
plt.show()

for lag in [20, 100, 300, 500]:
    print(f"Hurst exponents with {lag} lags ----")
    for column in df.columns:
        print(f"{column}: {get_hurst_exponent(df[column].values, lag):.4f}")
apple_df = yf.download("AAPL", 
                     start="2008-01-01", 
                     end="2023-05-31", 
                     progress=False)
ibm_df = yf.download("IBM", 
                     start="2008-01-01", 
                     end="2023-05-31", 
                     progress=False)
jpchase_df = yf.download("JPM", 
                     start="2008-01-01", 
                     end="2023-05-31", 
                     progress=False)
plt.plot(apple_df['Adj Close'],label ="Apple Inc.",color='g')
plt.plot(ibm_df['Adj Close'],label ="IBM",color='black')
plt.plot(jpchase_df['Adj Close'],label ="JPMorgan Chase & Co.",color='r')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Stock Price')
plt.legend()
plt.show()

for lag in [20, 100, 300, 500, 1000]:
    hurst_exp = get_hurst_exponent(apple_df["Adj Close"].values, lag)
    print(f"Hurst exponent for Apple Inc. with {lag} lags: {hurst_exp:.4f}")
for lag in [20, 100, 300, 500, 1000]:
    hurst_exp = get_hurst_exponent(ibm_df["Adj Close"].values, lag)
    print(f"Hurst exponent for IBM with {lag} lags: {hurst_exp:.4f}")
for lag in [20, 100, 300, 500, 1000]:
    hurst_exp = get_hurst_exponent(jpchase_df["Adj Close"].values, lag)
    print(f"Hurst exponent for JPMorgan Chase & Co. with {lag} lags: {hurst_exp:.4f}")
USTYield_df = yf.download("^TNX", 
                     start="2008-01-01", 
                     end="2023-05-31", 
                     progress=False)
plt.plot(USTYield_df['Adj Close'],label ="Treasury 10 Years Yield",color='g')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Treasury Yield 10 Years')
plt.legend()
plt.show()
for lag in [20, 100, 300, 500, 1000]:
    hurst_exp = get_hurst_exponent(USTYield_df["Adj Close"].values, lag)
    print(f"Hurst exponent for US Treasury 10 years yield with {lag} lags: {hurst_exp:.4f}")
