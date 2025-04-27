import pandas as pd
import numpy as np
import yfinance as yf
import scipy.optimize as sco
import plotly.graph_objs as go


def get_stock_data(symbols, start_date, end_date):
    """
    Загрузка котировок акций с Yahoo.

    symbols: список тикеров акций
    start_date: начальная дата в формате 'YYYY-MM-DD'
    end_date: конечная дата в формате 'YYYY-MM-DD'

    Возвращает DataFrame с ценами закрытия.
    """
    data = yf.download(symbols, start=start_date, end=end_date)['Close']
    return data


def calculation_portfolio_result(weights, mean_returns, cov_matrix):
    """
    Расчет доходности и волатильности портфеля.

    weights: массив с весами активов
    mean_returns: средние доходности активов
    cov_matrix: ковариационная матрица

    Возвращает ожидаемую доходность и стандартное отклонение портфеля.
    """
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_std_dev


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    """
    Отрицательное значение коэффициента Шарпа (для минимизации).

    Используется в функции оптимизации для нахождения портфеля
    с максимальным коэффициентом Шарпа.
    """
    p_ret, p_var = calculation_portfolio_result(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var


def get_portfolio_vol(weights, mean_returns, cov_matrix):
    """
    Расчет волатильности портфеля.

    weights: массив с весами активов
    mean_returns: средние доходности активов
    cov_matrix: ковариационная матрица

    Возвращает волатильность портфеля.
    """
    return calculation_portfolio_result(weights, mean_returns, cov_matrix)[1]


def find_max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    """
    Поиск портфеля с максимальным коэффициентом Шарпа.
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    opts = sco.minimize(negative_sharpe_ratio, num_assets * [1. / num_assets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return opts


def find_min_variance(mean_returns, cov_matrix):
    """
    Поиск портфеля с минимальной волатильностью.
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    opts = sco.minimize(get_portfolio_vol, num_assets * [1. / num_assets], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return opts


def find_efficient_return(mean_returns, cov_matrix, target_return):
    """
    Поиск портфеля с заданной доходностью и минимальной волатильностью.
    """
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
    """
    Расчет эффективной границы.

    range_of_returns: массив целевых доходностей
    Возвращает список оптимальных портфелей для каждого уровня доходности.
    """
    return [find_efficient_return(mean_returns, cov_matrix, ret) for ret in range_of_returns]


# Основной скрипт

stocks = ['CME', 'BANF', 'IDXX', 'NVDA']
num_assets = len(stocks)
start = '2023-01-01'
end = '2023-12-31'

# Загрузка дэйты
data = get_stock_data(stocks, start, end)

# Базовые расчёты
risk_free_rate = 0.0021
dur = 20
num_periods_annually = 252.0 / dur
windowed_data = data[::dur]
rets = np.log(windowed_data / windowed_data.shift(1)).dropna()
mean_daily_return = rets.mean()
covariance = rets.cov()

# Монте-Карло симуляция
num_portfolios = 5000
results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    weights_record.append(weights)
    pret, pvar = calculation_portfolio_result(weights, mean_daily_return, covariance)
    results[0, i] = pret * num_periods_annually
    results[1, i] = pvar * np.sqrt(num_periods_annually)
    results[2, i] = (results[0, i] - risk_free_rate) / results[1, i]

# Эффективная граница
target_returns = np.linspace(0.09, 0.26, 50) / (252. / dur)
efficient_portfolios = find_efficient_frontier(mean_daily_return, covariance, target_returns)

# Рисуем интерактивный график
trace_mc = go.Scatter(
    x=results[1, :],
    y=results[0, :],
    mode='markers',
    marker=dict(
        color=results[2, :],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Sharpe Ratio'),
        size=5,
        opacity=0.7
    ),
    name='Monte Carlo Portfolios'
)

trace_efficient = go.Scatter(
    x=[p['fun'] * np.sqrt(num_periods_annually) for p in efficient_portfolios],
    y=target_returns * num_periods_annually,
    mode='lines+markers',
    name='Efficient Frontier',
    marker=dict(color='black', symbol='x')
)

max_sharpe = find_max_sharpe_ratio(mean_daily_return, covariance, risk_free_rate)
rp, sdp = calculation_portfolio_result(max_sharpe['x'], mean_daily_return, covariance)
trace_max_sharpe = go.Scatter(
    x=[sdp * np.sqrt(num_periods_annually)],
    y=[rp * num_periods_annually],
    mode='markers',
    name='Max Sharpe Ratio',
    marker=dict(color='red', size=12, symbol='star')
)

min_var = find_min_variance(mean_daily_return, covariance)
rp, sdp = calculation_portfolio_result(min_var['x'], mean_daily_return, covariance)
trace_min_var = go.Scatter(
    x=[sdp * np.sqrt(num_periods_annually)],
    y=[rp * num_periods_annually],
    mode='markers',
    name='Min Variance',
    marker=dict(color='yellow', size=12, symbol='diamond')
)

layout = go.Layout(
    title='Portfolio Optimization: Efficient Frontier and Monte Carlo Simulation',
    xaxis=dict(title='Expected Volatility'),
    yaxis=dict(title='Expected Return'),
    hovermode='closest'
)

fig = go.Figure(data=[trace_mc, trace_efficient, trace_max_sharpe, trace_min_var], layout=layout)
fig.show()
