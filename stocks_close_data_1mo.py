import yfinance as yf
import pandas as pd

# Код использовался для импорта данных о доходностях по активам при
# отборе активов по отраслям, в качестве тикеров же указаны итоговые, так как
# при анализе было использовано около 50 тикеров, а в данном случае они
# не пригодятся


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
