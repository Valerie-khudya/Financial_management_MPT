import yfinance as yf
import pandas as pd


tickers = [
        'AAPL', 'AVGO', 'CME', 'CSCO', 'EQIX', 'GOOGL', 'IRM', 'ISRG',
        'NVDL', 'QQQ', 'SPY'
        ]

# Создание Excel таблицы
with pd.ExcelWriter("stocks_close_1mo.xlsx") as writer:
    for ticker in tickers:
        data = yf.download(
            ticker, start="2022-01-01", end="2024-12-31", interval="1mo"
            )
        data.to_excel(writer, sheet_name=ticker)

print("Готово!")
