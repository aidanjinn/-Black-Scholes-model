import os
from datetime import datetime
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import numpy as np
from numpy.ma.extras import average
from scipy.stats import norm
import yfinance as yf
from fredapi import Fred
from scipy.optimize import brentq

load_dotenv()
fred_api_key = os.getenv("FRED_API_KEY")

class Model:
    """
        S : Asset Price
        K : Option Strike Price
        T : Time to Expire in years
        r : Risk Free Interest Rate
        sigma : Volatility
    """
    def __init__(self, _S, _K, _T, _r, _sigma, _option_type = 'call'):
        self.S = float(_S)
        self.K = float(_K)
        self.T = float(_T)
        self.r = float(_r)
        self.sigma = float(_sigma)
        self.option_type = _option_type.lower()

        # '''
        #     Needed For Options Price
        # '''
        # self.d1 = (np.log(self.S / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
        # self.d2 = (self.d1 - self.sigma * np.sqrt(self.T))
        if self.sigma <= 0 or self.T <= 0:
            self.d1 = self.d2 = 0.0  # Neutralize
        else:
            self.d1 = (np.log(self.S / self.K) + (self.r + self.sigma ** 2 / 2) * self.T) / (
                        self.sigma * np.sqrt(self.T))
            self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

    def option_price(self):

        if self.option_type == 'call':
            return self.S * norm.cdf(self.d1) - self.K * (np.exp(-self.r * self.T)) * norm.cdf(self.d2)
        elif self.option_type == 'put':
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)
        else:
            raise ValueError('Option Type Error')



def calculate_sigma(stock_data, window = 252):
    log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    daily_volatility = log_returns.dropna().std()
    annualized_volatility = daily_volatility * np.sqrt(window)
    return annualized_volatility


fred = Fred(api_key=fred_api_key)

# Maturity codes:
maturity_codes = {
    1: 'DGS1',
    2: 'DGS2',
    3: 'DGS3',
    5: 'DGS5',
    7: 'DGS7',
    10: 'DGS10'
}

yields = []
maturities = []

for m, series_id in maturity_codes.items():
    try:
        series = fred.get_series_latest_release(series_id).dropna()
        value = float(series.iloc[-1])
        maturities.append(m)
        yields.append(value)
    except Exception as e:
        print(f"Error fetching yield for maturity {m}: {e}")

def interpolate_yield(maturities, yields, target_maturity):
    return float(np.interp(target_maturity, maturities, yields))

def implied_volatility(market_price, S, K, T, r, option_type, fallback_sigma=None):
    def objective(sigma):
        try:
            model = Model(S, K, T, r, sigma, option_type)
            return model.option_price() - market_price
        except Exception:
            return np.nan

    try:
        return brentq(objective, 1e-6, 5.0)
    except Exception:
        return fallback_sigma if fallback_sigma is not None else np.nan


while True:

    while True:
        try:
            SYMBOL = input("Enter Stock Ticker (e.g., AAPL, MSFT): ").upper().strip()
            OPTION = input("Enter Option Type (CALL OR PUT): ").lower().strip()

            if SYMBOL.lower() == 'exit' or OPTION.lower() == 'exit':
                exit()

            if OPTION != 'call' and OPTION != 'put':
                print('Invalid Option Type')
                continue

            if not SYMBOL:
                print("Please enter a stock ticker.")
                continue

            ticker = yf.Ticker(SYMBOL)
            expiry_dates = ticker.options[:3]

            if not expiry_dates:
                print(f"No options data available for {SYMBOL}. Please try another ticker.")
                continue

            stock_data = yf.download(SYMBOL, period="5d", interval="1m", progress=False)

            if stock_data.empty:
                print(f"No price data available for {SYMBOL}. Please try another ticker.")
                continue
            else:
                break

        except Exception as e:
            print(f"Error occurred: {str(e)}. Please try again.")

    sigma = float(calculate_sigma(stock_data))
    option_chain = ticker.option_chain(expiry_dates[0])

    if OPTION == 'call':
        options_df = option_chain.calls
    else:
        options_df = option_chain.puts

    expiry_date_obj = datetime.strptime(expiry_dates[0], '%Y-%m-%d')
    today = datetime.now()
    time_diff = expiry_date_obj - today
    time_to_expiry_years = time_diff.total_seconds() / (365.25 * 24 * 60 * 60)

    sample_call = options_df.iloc[0]

    S = float(stock_data['Close'].iloc[-1].item())
    K = sample_call['strike']
    T = time_to_expiry_years
    r = interpolate_yield(maturities, yields, T) / 100
    option_type = OPTION

    options_sample = options_df[np.abs(options_df['strike'] - S) < 20].head(40)

    bs_prices = []
    market_prices = []
    strikes = []
    diff = []

    for _, row in options_sample.iterrows():
        K = float(row['strike'])
        market_price = float(row['lastPrice'])

        if np.isnan(market_price) or market_price == 0:
            continue

        expiry_date_obj = datetime.strptime(expiry_dates[0], '%Y-%m-%d')
        today = datetime.now()

        if T <= 0:
            continue

        implied_sigma = implied_volatility(market_price, S, K, T, r, option_type, fallback_sigma=sigma)

        model = Model(_S=S, _K=K, _T=T, _r=r, _sigma=implied_sigma, _option_type= option_type)
        bs_price = float(model.option_price())

        print(f"Strike: {K:.1f}, Predicted Price: {bs_price:.2f}, Market Price: {market_price:.2f}")

        strikes.append(K)
        market_prices.append(market_price)
        bs_prices.append(bs_price)
        diff.append(abs(market_price - bs_price))

    print(f"Average Difference in Model Price and Market: {average(diff):.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(strikes, market_prices, label='Market Price', marker='o')
    plt.plot(strikes, bs_prices, label='Black-Scholes Price', marker='x')
    plt.xlabel('Strike Price')
    plt.ylabel('Option Price')
    plt.title(f'{SYMBOL} Option Prices ({expiry_dates[0]})')
    plt.legend()
    plt.grid(True)
    plt.show()


