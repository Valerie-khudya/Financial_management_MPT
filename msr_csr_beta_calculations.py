from decimal import Decimal
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import skew, kurtosis, norm
from pypfopt import expected_returns, risk_models


def get_data(tickers, start, end):
    """Загрзука данных цен закрытия из Yahoo"""
    data = yf.download(tickers, start=start, end=end)['Close'].dropna()
    nasdaq_data = yf.download('^IXIC', start=start, end=end)['Close'].pct_change().dropna()
    nasdaq_data = nasdaq_data.iloc[:, 0]
    return data, nasdaq_data


def get_returns_and_cov_matrix(data):
    """Доходности и ковариационная матрица"""
    expected_annual_return = expected_returns.mean_historical_return(data)
    cov_matrix = risk_models.sample_cov(data)
    daily_returns = data.pct_change().dropna()
    return expected_annual_return, cov_matrix, daily_returns


def calculate_csr(
        portfolio_daily_ret, portf_mean, daily_rf, alpha
        ):
    """Расчет кондиционного коэффициента Шарпа для портфеля"""
    var = np.quantile(portfolio_daily_ret, alpha)
    cvar = portfolio_daily_ret[portfolio_daily_ret <= var].mean()
    csr = (portf_mean - daily_rf) / abs(cvar)
    return csr


def calculate_msr(
        z, portfolio_daily_ret, portf_mean, daily_rf, portf_std,
        ):
    """Расчет модифицированного коэффициента Шарпа для портфеля"""
    portf_skew = skew(portfolio_daily_ret)
    portf_kurtosis = kurtosis(portfolio_daily_ret, fisher=True)

    z_mvar = (
        - z
        + (1 / 6) * (z**2 - 1) * portf_skew
        + (1 / 24) * (z**3 - 3 * z) * portf_kurtosis
        - (1 / 36) * (2 * z**3 - 5 * z) * (portf_skew**2)
    )
    if z_mvar * portf_std != 0:
        modified_sharpe = (portf_mean - daily_rf) / (portf_mean - z_mvar * portf_std)
        msr = modified_sharpe
    return msr


def calculate_beta(portf_returns, market_returns):
    """Расчет беты портфеля"""
    combined = pd.DataFrame({
        'portfolio': portf_returns,
        'Market': market_returns
    }).dropna()
    covariance = np.cov(combined['portfolio'], combined['Market'])[0, 1]
    market_variance = np.var(combined['Market'])
    beta = covariance / market_variance
    return beta


if __name__ == '__main__':
    # Константы
    RISK_FREE_RATE_2023 = 0.0469
    RISK_FREE_RATE_2024 = 0.0419
    ALPHA = 0.01

    # Параметры, связанные с константами
    daily_rf_2023 = RISK_FREE_RATE_2023 / 252
    daily_rf_2024 = RISK_FREE_RATE_2024 / 252
    z = norm.ppf(1 - ALPHA)

    tickers = [
        'AAPL', 'AVGO', 'CME', 'CSCO', 'EQIX', 'GOOGL', 'IRM', 'ISRG',
        'NVDL', 'QQQ', 'SPY'
        ]

    # Загрузка данных
    p_data_2023, nasdaq_data_2023 = get_data(tickers, '2023-01-01', '2023-12-31')
    (expected_annual_return_2023, cov_matrix_2023,
     daily_returns_2023) = get_returns_and_cov_matrix(p_data_2023)

    p_data_2024, nasdaq_data_2024 = get_data(tickers, '2024-01-01', '2024-12-31')
    (expected_annual_return_2024, cov_matrix_2024,
     daily_returns_2024) = get_returns_and_cov_matrix(p_data_2024)

    all_weights = {}
    with open('all_weights.txt', 'r', encoding='utf-8') as infile:
        for line in infile:
            if '-- ' in line:
                name = line.strip()
                weights_local = []
            elif len(line) > 2:
                weight = float(line.strip().split(': ')[1])
                weights_local.append(weight)
            else:
                continue
            if len(weights_local) == 11:
                all_weights[name] = np.array(weights_local)

    for rate, weights in all_weights.items():
        if '-- Максимальный коэффиент Шарпа:' in rate:
            max_sharpe_weights = weights
        daily_returns = daily_returns_2023.dot(weights)
        portf_mean = daily_returns.mean()
        portf_std = daily_returns.std()
        ret = np.dot(weights, expected_annual_return_2023)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_2023, weights)))

        csr = Decimal(calculate_csr(
            daily_returns, portf_mean, daily_rf_2023, ALPHA)
            )

        msr = Decimal(calculate_msr(
            z, daily_returns, portf_mean, daily_rf_2023, portf_std
             ))

        beta = Decimal(calculate_beta(daily_returns, nasdaq_data_2023))

        print('\n' + rate)
        print(f'Кондиционный коэффициент Шарпа: {csr:.4}')
        print(f'Модифицированный коэффициент Шарпа: {msr:.4}')
        print(f'Бета портфеля: {beta:.4}')

    # Расчеты для выбранного портфеля за 2024 г.
    daily_returns_2024 = daily_returns_2024.dot(max_sharpe_weights)
    portf_mean_2024 = daily_returns_2024.mean()
    portf_std_2024 = daily_returns_2024.std()
    ret_2024 = np.dot(max_sharpe_weights, expected_annual_return_2024)
    vol_2024 = np.sqrt(np.dot(max_sharpe_weights.T, np.dot(
        cov_matrix_2024, max_sharpe_weights))
        )

    csr_2024 = Decimal(calculate_csr(
        daily_returns_2024, portf_mean_2024, daily_rf_2024, ALPHA)
        )

    msr_2024 = Decimal(calculate_msr(
        z, daily_returns_2024, portf_mean_2024, daily_rf_2024, portf_std_2024
         ))

    beta_2024 = Decimal(calculate_beta(daily_returns_2024, nasdaq_data_2024))

    print('\n' + '-- Максимальный Коэффициент Шарпа (выбранный портфель, 2024 г.)')
    print(f'Кондиционный коэффициент Шарпа: {csr_2024:.4}')
    print(f'Модифицированный коэффициент Шарпа: {msr_2024:.4}')
    print(f'Бета портфеля: {beta_2024:.4}')
