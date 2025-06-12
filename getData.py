import yfinance as yf

data = yf.download("^IXIC", start="1990-01-01", end="2024-12-31")
data.to_csv('nasdaq_1990_2024.csv')