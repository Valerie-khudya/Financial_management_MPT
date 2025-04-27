import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.optimize as sco


def get_stock_data(symbols, start_date, end_date):
    """Загрузка котировок акция с Yahoo"""
    data = yf.download(symbols, start=start_date, end=end_date)['Close']
    return data


def calculation_portfolio_result(weights, mean_returns, cov_matrix):
    """Расчет доходности и волотильности портфеля"""
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_std_dev


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """Отрицательное значение коэф. Шарпа (для минимизации)"""
    # Сравнить с требованиями по проекту
    p_ret, p_std_dev = calculation_portfolio_result(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_std_dev


def get_portfolio_vol(weights, mean_returns, cov_matrix):
    """расчет волотильности портфеля"""
    return calculation_portfolio_result(weights, mean_returns, cov_matrix)[1]


def find_max_Sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    """Нахождение портфеля с мин. коэф. Шарпа"""
    # аналогично функции negative_sharpe_ratio()
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    opts = sco.minimize(negative_sharpe_ratio, num_assets * [1. / num_assets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return opts


def find_min_variance(mean_returns, cov_matrix):
    """Поиск портфеля с минимальной ковариацией"""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    opts = sco.minimize(get_portfolio_vol, num_assets * [1. / num_assets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return opts


def find_efficient_return(mean_returns, cov_matrix, target_return):
    """Поиск портфеля с заданным уровнем доходности и мин. волатильности"""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def get_portfolio_return(weights):
        return calculation_portfolio_result(weights, mean_returns, cov_matrix)[0]

    constraints = (
        {'type': 'eq', 'fun': lambda x: get_portfolio_return(x) - target_return},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    )
    bounds = tuple((0, 1) for _ in range(num_assets))

    return sco.minimize(get_portfolio_vol, num_assets * [1. / num_assets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)


def find_efficient_frontier(mean_returns, cov_matrix, range_of_returns):
    """Расчет эффективной границы"""
    return [find_efficient_return(mean_returns, cov_matrix, ret) for ret in range_of_returns]


# -- Запуск формирования портфеля

stocks = ['CME', 'BANF', 'IDXX', 'NVDA']
num_assets = len(stocks)
start = '2023-01-01'
end = '2023-12-31'

# Загрузка данных
data = get_stock_data(stocks, start, end)

# Расчёты
risk_free_rate = 0.0021
dur = 20
num_periods_annually = 252.0 / dur
windowed_data = data[::dur]
rets = np.log(windowed_data / windowed_data.shift(1))
mean_daily_return = rets.mean()
covariance = rets.cov()

# Монте-Карло симуляция
# Необходимо сравнить с требованиями в задании
plt.figure(figsize=(10, 7))
num_portfolios = 100000
results = np.zeros((3, num_portfolios))

for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    expected_return, expected_variance = calculation_portfolio_result(weights, mean_daily_return, covariance)

    # Расчет коэф. Шарпа и годовых значений доходности и волотильности
    results[0, i] = expected_return * num_periods_annually
    results[1, i] = expected_variance * np.sqrt(num_periods_annually)
    results[2, i] = (results[0, i] - risk_free_rate) / results[1, i]

# Результат Монте-Карло симуляции
plt.scatter(results[1, :], results[0, :], c=results[2, :], marker='o')

# Эффективная граница, при годовой доходости от 0,09 до 1,2 (границы линии по оси Y)
target_returns = np.linspace(0.09, 1.2, 150) / num_periods_annually
efficient_portfolios = find_efficient_frontier(mean_daily_return, covariance, target_returns)
plt.plot(
    [p['fun'] * np.sqrt(num_periods_annually) for p in efficient_portfolios],
    target_returns * num_periods_annually, marker='x'
)

# Портфель с максимальным коэффициентом Шарпа
max_Sharpe = find_max_Sharpe_ratio(mean_daily_return, covariance, risk_free_rate)
rp, sdp = calculation_portfolio_result(max_Sharpe['x'], mean_daily_return, covariance)
plt.plot(sdp * np.sqrt(num_periods_annually), rp * num_periods_annually, 'r*', markersize=15.0, label='Макс. коэф. Шарпа')

# Портфель с минимальной ковариацией
min_var = find_min_variance(mean_daily_return, covariance)
rp, sdp = calculation_portfolio_result(min_var['x'], mean_daily_return, covariance)
plt.plot(sdp * np.sqrt(num_periods_annually), rp * num_periods_annually, 'y*', markersize=15.0, label='Мин. волатильность')

# Равновзвешенный портфель
equal_weights = np.array([1/num_assets] * num_assets)
rp_eq, sdp_eq = calculation_portfolio_result(equal_weights, mean_daily_return, covariance)
rp_eq_annual = rp_eq * num_periods_annually
sdp_eq_annual = sdp_eq * np.sqrt(num_periods_annually)
plt.plot(sdp_eq_annual, rp_eq_annual, 'k*', markersize=15.0, label='Равновзвешенный')

# Отображение названия каждой "звездочки"
plt.legend(loc='upper left')

plt.grid(True)
plt.xlabel('Ожидаемая волотильность')
plt.ylabel('Ожидаемая доходность')
plt.colorbar(label='Коэф. Шарпа')
plt.title('Портфели с несколькими активами Portfolios')
plt.tight_layout()
plt.savefig('монте_карло_симуляция_для_портфолио.png', dpi=100)

# Сравнение портфелей
ind = np.arange(num_assets)
width = 0.35
fig, ax = plt.subplots(figsize=(8, 6))
rects_1 = ax.bar(ind, max_Sharpe['x'], width, color='r', alpha=0.75)
rects_2 = ax.bar(ind + width, min_var['x'], width, color='b', alpha=0.75)
ax.set_ylabel('Веса активов в портфеле')
ax.set_ylim(0, 0.9)
ax.set_title('Сравнение разных комбинаций портфеля')
ax.set_xticks(ind + width)
ax.set_xticklabels(stocks)
plt.tight_layout()
ax.legend((rects_1[0], rects_2[0]), ('Макс. коэф. Шарпа', 'Мин. волатильность'))
plt.savefig('portfolio_compositions.png', dpi=100)
plt.show()
