import yfinance as yf
import pandas as pd


# Список тикеров
tickers = ['TMUS', 'CHTR', 'WBD', 'CSCO', 'ROKU', 'EXTR', 'TIGO', 'IRDM']

# Создание Excel таблицы
with pd.ExcelWriter("multi_stock_data.xlsx") as writer:
    for ticker in tickers:
        data = yf.download(
            ticker, start="2021-01-01", end="2025-01-01", interval="1mo"
            )
        data.to_excel(writer, sheet_name=ticker)

print("Данные сохранены в файл, он сохранился в этот же проект (папку)")
