import yfinance as yf
import pandas as pd


def get_close_to_excel(tickers, start, end):
    # Загрузка только цен закрытия по тикерам
    data = yf.download(tickers, start=start, end=end)['Close']
    nasdaq_data = yf.download(
        nasdaq_ticker, start=start, end=end
        )['Close']

    # Преобразование индекса в строку даты
    data.index = data.index.strftime('%Y-%m-%d')
    nasdaq_data.index = nasdaq_data.index.strftime('%Y-%m-%d')

    # Добавление столбцов День и Дата
    data.insert(0, 'Дата', data.index)
    data.insert(0, 'День', range(1, len(data) + 1))

    # Разделительная пустая колонка между данными по индексу NASDAQ и активов
    data[''] = ''

    # Добавление индекса NASDAQ (composite)
    data['NASDAQ composite (^IXIC)'] = nasdaq_data.reindex(data['Дата']).values
    return data


tickers = [
        'AAPL', 'AVGO', 'CME', 'CSCO', 'EQIX', 'GOOGL', 'IRM', 'ISRG',
        'NVDL', 'QQQ', 'SPY'
        ]
nasdaq_ticker = '^IXIC'

data_2022 = get_close_to_excel(tickers, '2022-01-01', '2022-12-31')
data_2023 = get_close_to_excel(tickers, '2023-01-01', '2023-12-31')
data_2024 = get_close_to_excel(tickers, '2024-01-01', '2024-12-31')

# Сохранение данных активов в Excel
output_filename = 'close_2022-2024.xlsx'
with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    data_2022.to_excel(writer, sheet_name='Close data_2022', index=False)
    data_2023.to_excel(writer, sheet_name='Close data_2023', index=False)
    data_2024.to_excel(writer, sheet_name='Close data_2024', index=False)


# Безрисковая доходность (данные казначейства США)
# Перед запускам важно отметить, что в этой папке есть данные
# daily-treasury-rates, но если вы решите запускать код
# не через эту папку, а каким-то другим путем, то убедитесь, что этот файл
# находится в нужной папке, иначе будет ошибка.
try:
    yield_data = pd.read_csv('daily-treasury-rates.csv')

    # Добавление колонки Yield Curve Name
    yield_data.insert(0, 'Yield Curve Name', 'US Treasury')

    # Добавление данных по treasury rates на второй лист того же файла
    with pd.ExcelWriter(
        output_filename, engine='openpyxl', mode='a'
    ) as writer:
        yield_data.to_excel(
            writer, sheet_name='Безрисковая доходность', index=False
            )

    print("Файл сохранен!")

except FileNotFoundError:
    print("Файла 'daily-treasury-rates.csv' в этой папке нет")
