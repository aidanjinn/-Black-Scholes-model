# Black-Scholes Option Pricing Model

A Python implementation of the Black-Scholes model for pricing European call and put options. This project fetches real-world stock and options data using Yahoo Finance (`yfinance`) and risk-free interest rates from the FRED API.

## Features
- Calculates option prices using the Black-Scholes formula
- Fetches real-time stock data and options chains
- Estimates implied volatility from market prices
- Compares model prices with market prices
- Visualizes results using Matplotlib

## Technologies Used
- Python 3.12
- Libraries:
  - `numpy` for numerical computations
  - `scipy` for statistical functions and root-finding
  - `yfinance` for stock and options data
  - `fredapi` for risk-free interest rates
  - `matplotlib` for visualization
  - `python-dotenv` for environment variables

## How It Works
1. Takes user input for stock ticker and option type (call/put)
2. Fetches:
   - Stock price data from Yahoo Finance
   - Options chain data
   - Risk-free rates from FRED (US Treasury yields)
3. Calculates:
   - Historical volatility
   - Time to expiration
   - Option prices using Black-Scholes formula
   - Implied volatility
4. Compares model prices with market prices
5. Plots the results

## Black-Scholes Model
The Black-Scholes model is a mathematical model for pricing options contracts. The key formulas are:

**Call Option Price:**
C = S * N(d1) - K * e^(-rT) * N(d2)

**Put Option Price:**
P = K * e^(-rT) * N(-d2) - S * N(-d1)

Where:
- S = Current stock price
- K = Strike price
- T = Time to expiration (in years)
- r = Risk-free interest rate
- σ = Volatility
- N() = Cumulative standard normal distribution
- d1, d2 = Intermediate calculations

## Usage
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your FRED API key: `FRED_API_KEY=your_key_here`
4. Run the script: `python black_scholes.py`
5. Follow the prompts to enter stock ticker and option type

