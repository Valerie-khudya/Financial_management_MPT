import os
import sys
from decimal import Decimal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis, norm
from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting


def get_data(tickers, start, end):
    """Загрузка данных цен закрытия из Yahoo"""
    data = yf.download(tickers, start=start, end=end)['Close'].dropna()
    return data


def get_returns_and_cov_matrix(data):
    """Доходности и ковариационная матрица"""
    expected_annual_return = expected_returns.mean_historical_return(data)
    cov_matrix = risk_models.sample_cov(data)
    daily_returns = data.pct_change().dropna()
    return expected_annual_return, cov_matrix, daily_returns


def generate_portfolios(
        num_portfolios, expected_return, cov_matrix, rf, bounds
        ):
    """Генерация n-го количества портфелей"""
    # Инициализация записи весов активов и массива,
    # в который будут записываться данные по сформированным портфелям.
    results = np.zeros((3, num_portfolios))
    weights_record = []

    # Генерация портфелей ("облака результатов")
    for i in range(num_portfolios):
        low_bound, high_bound = zip(*bounds)
        while True:
            weights = np.random.uniform(low_bound, high_bound)
            weights /= weights.sum()
            if all(low_b <= w <= high_b for w, low_b, high_b in zip(
                weights, low_bound, high_bound
            )):
                break
        weights_record.append(weights)

        ret = np.dot(weights, expected_return)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - rf) / vol

        results[0, i] = vol
        results[1, i] = ret
        results[2, i] = sharpe
    return results, weights_record


def find_min_volatility_portfolio(
        tickers, expected_return, cov_matrix, min_target_return, bounds
        ):
    """
    Поиск портфеля с минимальной волатильностью при
    заданной минимально требуемой доходности
    """
    n = len(tickers)
    w0 = np.ones(n) / n

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: np.dot(w, expected_return) - (
            min_target_return)}
    ]

    def calculate_portfolio_volatility(w, cov_matrix):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

    result = minimize(calculate_portfolio_volatility, w0, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints
                      )

    min_vol = result.fun
    min_vol_weights = pd.Series(result.x, index=tickers)
    min_vol_ret = np.dot(min_vol_weights, expected_return)
    return min_vol, min_vol_weights, min_vol_ret


def find_max_msr(
        tickers, daily_returns, daily_rf,
        bounds, expected_return, cov_matrix, z
        ):
    """Поиск портфеля с максимальным модифицированным коэффициентом Шарпа"""
    n = len(tickers)
    w0 = np.ones(n) / n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    def calculate_msr(w, daily_returns, daily_rf, z):
        portfolio_daily_ret = daily_returns @ w
        portfolio_daily_std = portfolio_daily_ret.std()
        portfolio_skew = skew(portfolio_daily_ret)
        portfolio_kurtosis = kurtosis(portfolio_daily_ret, fisher=True)
        z_mvar = (
            - z
            + (1 / 6) * (z**2 - 1) * portfolio_skew
            + (1 / 24) * (z**3 - 3 * z) * portfolio_kurtosis
            - (1 / 36) * (2 * z**3 - 5 * z) * (portfolio_skew**2)
        )
        if z_mvar * portfolio_daily_std != 0:
            mvar = portfolio_daily_ret.mean() + z_mvar * portfolio_daily_std
            modified_sharpe = (
                (portfolio_daily_ret.mean() - daily_rf) / abs(mvar)
                )
        else:
            modified_sharpe = 0
        return - modified_sharpe

    result = minimize(calculate_msr, w0,
                      args=(daily_returns, daily_rf, z),
                      method='SLSQP', bounds=bounds, constraints=constraints
                      )

    max_msr = - result.fun
    max_msr_weights = pd.Series(result.x, index=tickers)
    max_msr_ret = np.dot(max_msr_weights, expected_return)
    max_msr_vol = np.sqrt(
        np.dot(max_msr_weights.T, np.dot(cov_matrix, max_msr_weights))
        )
    return max_msr, max_msr_weights, max_msr_ret, max_msr_vol


def find_max_csr(
        tickers, daily_returns, daily_rf, alpha, bounds,
        expected_return, cov_matrix
        ):
    """Поиск портфеля с максимальным кондиционным коэффициентом Шарпа"""
    n = len(tickers)
    w0 = np.ones(n) / n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    def calculate_csr(w, daily_returns, daily_rf, alpha):
        portfolio_daily_ret = daily_returns @ w
        var = np.quantile(portfolio_daily_ret, alpha)
        cvar = portfolio_daily_ret[portfolio_daily_ret <= var].mean()
        if cvar >= 0:
            return sys.maxsize
        conditional_sharpe = (
            (portfolio_daily_ret.mean() - daily_rf) / abs(cvar)
            )
        return - conditional_sharpe

    result = minimize(calculate_csr, w0,
                      args=(daily_returns, daily_rf, alpha),
                      method='SLSQP', bounds=bounds, constraints=constraints
                      )
    max_csr_weights = pd.Series(result.x, index=tickers)
    max_csr = - result.fun
    max_csr_ret = np.dot(max_csr_weights, expected_return)
    max_csr_vol = np.sqrt(
        np.dot(max_csr_weights.T, np.dot(cov_matrix, max_csr_weights))
        )
    return max_csr, max_csr_weights, max_csr_ret, max_csr_vol


def print_and_save(text, file='all_weights.txt'):
    print(text)
    with open(file, 'a', encoding='utf-8') as outfile:
        outfile.write(str(text) + '\n')


if __name__ == '__main__':
    # Константы
    NUM_PORTFOLIOS = 100000
    RISK_FREE_RATE = 0.0469
    ALPHA = 0.01
    MIN_VOLATILITY_PORTFOLIO_RETURN = 0.4448
    MAX_RETURN_PORTFOLIO_VOLATILITY = 0.1727

    # Параметры, связанные с константами
    daily_rf = RISK_FREE_RATE / 252
    z = norm.ppf(1 - ALPHA)

    tickers = [
        'AAPL', 'AVGO', 'CME', 'CSCO', 'EQIX', 'GOOGL', 'IRM', 'ISRG',
        'NVDL', 'QQQ', 'SPY'
        ]

    # Ограничения на веса активов
    bounds = [(0.02, 0.14)] * len(tickers)

    # Ограничения на ETF
    bounds[8] = (0.1, 0.15)  # NVDL
    bounds[9] = (0.1, 0.17)  # QQQ
    bounds[10] = (0.1, 0.19)  # SPY

    # Загрузка данных
    data_2023 = get_data(tickers, '2023-01-01', '2023-12-31')
    (expected_annual_return_2023, cov_matrix_2023,
     daily_returns_2023) = get_returns_and_cov_matrix(data_2023)

    # EF: Построение Границы эффективности
    ef = EfficientFrontier(expected_annual_return_2023, cov_matrix_2023,
                           weight_bounds=bounds
                           )
    fig, ax = plt.subplots(figsize=(10, 7))

    # 100000 portfolios: Генерация 'облака результатов'
    results, weights_record = generate_portfolios(
        NUM_PORTFOLIOS, expected_annual_return_2023, cov_matrix_2023,
        RISK_FREE_RATE, bounds
        )

    # EQ_W: Равновзвешенный портфель
    eq_weights = np.array([1/len(tickers)] * len(tickers))
    eq_ret = np.dot(eq_weights, expected_annual_return_2023)
    eq_vol = np.sqrt(np.dot(eq_weights.T, np.dot(cov_matrix_2023, eq_weights)))

    # MAX_RET: Портфель с максимальной доходностью
    # с заданной максимально возможной волатильностью
    ef_max_ret = EfficientFrontier(
        expected_annual_return_2023, cov_matrix_2023, weight_bounds=bounds
        )
    ef_max_ret.efficient_risk(
        target_volatility=MAX_RETURN_PORTFOLIO_VOLATILITY
        )
    max_ret, max_ret_vol, max_ret_sharpe = ef_max_ret.portfolio_performance()

    # MIN_VOL: Портфель с минимальной волатильностью
    # с заданной минимально требуемой доходностью
    min_vol, min_vol_weights, min_vol_ret = find_min_volatility_portfolio(
        tickers, expected_annual_return_2023, cov_matrix_2023,
        MIN_VOLATILITY_PORTFOLIO_RETURN, bounds
        )

    # MAX_SHARPE: Портфель с максимальным коэффициентом Шарпа
    ef_s = EfficientFrontier(
        expected_annual_return_2023, cov_matrix_2023, weight_bounds=bounds
        )
    ef_s.max_sharpe()
    (max_sharpe_ret, max_sharpe_vol,
     max_sharpe_sharpe) = ef_s.portfolio_performance()

    # MAX_MSR: Портфель с максимальным модифицированным коэффициентом Шарпа
    max_msr, max_msr_weights, max_msr_ret, max_msr_vol = find_max_msr(
        tickers, daily_returns_2023, daily_rf, bounds,
        expected_annual_return_2023, cov_matrix_2023, z
        )

    # MAX_CSR: Портфель с максимальным кондиционным коэффициентом Шарпа
    max_csr, max_csr_weights, max_csr_ret, max_csr_vol = find_max_csr(
        tickers, daily_returns_2023, daily_rf, ALPHA, bounds,
        expected_annual_return_2023, cov_matrix_2023
        )

    # Нахождение точки выбранного портфеля: MAX_SHARPE (2024 г)
    data_2024 = get_data(tickers, '2024-01-01', '2024-12-31')
    (expected_annual_return_2024, cov_matrix_2024,
     daily_returns_2024) = get_returns_and_cov_matrix(data_2024)

    portf_2024_weights = np.array(list(ef_s.clean_weights().values()))
    portf_2024_ret = np.dot(portf_2024_weights, expected_annual_return_2024)
    portf_2024_vol = np.sqrt(np.dot(
        portf_2024_weights.T, np.dot(cov_matrix_2024, portf_2024_weights
                                     )))

    # Если файл 'all_weights.txt' существует, то данные сотрутся
    # и запишутся заново, файл пригодится для работы кода из
    # 'msr_csr_beta_calculations.py', который запускается в ручную,
    # так как не всегда нужен
    if os.path.exists('all_weights.txt'):
        os.remove('all_weights.txt')

    # Вывод результатов в терминале и сохранение их в файл

    print_and_save('-- Равновзвешенный портфель:')
    print(f'Доходность: {eq_ret:.4f}, Риск: {eq_vol:.4f}')
    print('Веса равные для всех активов:')
    for ticker, weight in zip(tickers, eq_weights):
        print_and_save(f'{ticker}: {weight:.4f}')

    print_and_save('\n-- Максимальная доходность (с огр. на макс. риск):')
    print(f'Доходность: {max_ret:.4f}, Риск: {max_ret_vol:.4f}')
    print('Веса этого портфеля:')
    for ticker, weight in zip(tickers, ef_max_ret.clean_weights().values()):
        print_and_save(f'{ticker}: {weight:.4f}')

    print_and_save('\n-- Минимальная волатильность (с огр. на мин. доходность):')
    print(f'Доходность: {min_vol_ret:.4f}, Риск: {min_vol:.4f}')
    print('Веса этого портфеля:')
    for ticker, weight in min_vol_weights.items():
        print_and_save(f"{ticker}: {weight:.4f}")

    print_and_save(f'\n-- Максимальный коэффиент Шарпа: {max_sharpe_sharpe:.4f}')
    print('Веса этого портфеля:')
    for ticker, weight in zip(tickers, ef_s.clean_weights().values()):
        print_and_save(f'{ticker}: {weight:.3f}')

    print_and_save(f'\n-- Максимальный Модифицированный коэф. Шарпа: {max_msr:.4f}')
    print('Веса этого портфеля:')
    for ticker, weight in max_msr_weights.items():
        print_and_save(f"{ticker}: {weight:.4f}")

    print_and_save(f'\n-- Максимальный Кондиционный коэф. Шарпа: {max_csr:.4f}')
    print('Веса этого портфеля:')
    for ticker, weight in max_csr_weights.items():
        print_and_save(f"{ticker}: {weight:.4f}")

    # Визуализация
    plotting.plot_efficient_frontier(ef, ax=ax,
                                     show_assets=False)
    scatter = ax.scatter(
        results[0, :], results[1, :], c=results[2, :], cmap='viridis',
        alpha=0.4
        )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Коэффициент Шарпа')

    ax.scatter(
        eq_vol, eq_ret, marker='*', color='black', s=200,
        label='Равновзвешенный', alpha=0.7
        )
    ax.scatter(
        max_ret_vol, max_ret, marker='*', color='green', s=200,
        label='Максимальная доходность', alpha=0.7
        )
    ax.scatter(min_vol, min_vol_ret,
               marker='*', color='b', s=200,
               label='Минимальная волатильность', alpha=0.7
               )
    ax.scatter(max_sharpe_vol, max_sharpe_ret, marker='*', color='r', s=200,
               label='Макс. коэф. Шарп (выбранный портфель 2023 г.)', alpha=1
               )
    ax.scatter(
        max_msr_vol, max_msr_ret,
        marker='*', color='orange', s=200, label='Макс. Модиф. коэф. Шарпа',
        alpha=0.7
        )
    ax.scatter(
        max_csr_vol, max_csr_ret,
        marker='*', color='grey', s=200, label='Макс. Кондиц. коэф. Шарпа',
        alpha=0.7
        )
    ax.scatter(
        portf_2024_vol, portf_2024_ret, marker='*', color='magenta', s=180,
        label='Выбранный портфель (2024 г.)', alpha=1
        )

    ax.set_title('Граница эффективности')
    ax.set_xlabel('Волатильность')
    ax.set_ylabel('Доходность')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('efficient_frontier_nasdaq.png', dpi=300, bbox_inches='tight')
    plt.show()
