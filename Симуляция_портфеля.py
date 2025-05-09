import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models, EfficientFrontier, plotting
from scipy.stats import skew, kurtosis, norm


def get_data(tickers, start, end):
    """Загзрука данных цен закрытия из Yahoo"""
    data = yf.download(tickers, start=start, end=end)['Close'].dropna()
    return data


def get_returns_and_cov_matrix(data):
    """Доходности и ковариационная матрица"""
    expected_annual_return = expected_returns.mean_historical_return(data)
    cov_matrix = risk_models.sample_cov(data)
    daily_returns = data.pct_change().dropna()
    return expected_annual_return, cov_matrix, daily_returns


def find_max_csr(
        ret, vol, weights, portf_returns, portf_mean, daily_rf,
        max_csr, best_csr_ret, best_csr_vol, best_csr_weights, alpha
        ):
    """Поиск макисмального кондиционного коэф. Шарпа среди двух вариантов,
    которые прописаны как аргументы функции"""
    excess_returns = portf_returns - daily_rf  # Доходности с поправкой на безрисковую ставку в день.
    VaR = np.percentile(excess_returns, alpha * 100)  # пороговое значение α-квантили — то есть потеря в худшие 5% дней.
    tail_losses = excess_returns[excess_returns <= VaR]  # реальные доходности в "хвосте" распределения — те, что ниже или равны VaR

    CVaR = tail_losses.mean()
    csr = portf_mean / abs(CVaR)

    if csr > max_csr:
        max_csr, best_csr_ret, best_csr_vol = csr, ret, vol
        best_csr_weights = weights
    return max_csr, best_csr_ret, best_csr_vol, best_csr_weights


def find_max_modified_sharpe(
        z, portf_returns, portf_mean, daily_rf, portf_std,
        ret, vol, weights, max_msr, best_msr_ret, best_msr_vol,
        best_msr_weights
        ):
    """Поиск максимального модифицированного коэф. Шарпа среди двух вариантов,
    которые прописаны как аргументы функции"""
    portf_skew = skew(portf_returns)
    portf_kurtosis = kurtosis(portf_returns, fisher=True)

    z_cf = (
        z
        + (1 / 6) * (z**2 - 1) * portf_skew
        + (1 / 24) * (z**3 - 3 * z) * portf_kurtosis
        - (1 / 36) * (2 * z**3 - 5 * z) * (portf_skew**2)
    )
    if z_cf * portf_std != 0:
        modified_sharpe = (portf_mean - daily_rf) / (z_cf * portf_std)
        if modified_sharpe > max_msr:
            max_msr, best_msr_ret, best_msr_vol = modified_sharpe, ret, vol
            best_msr_weights = weights
    return max_msr, best_msr_ret, best_msr_vol, best_msr_weights


def generate_portfolios(num_portfolios, tickers, expected_return, cov_matrix,
                        rf, daily_rf, ALPHA, min_vol_return, daily_returns, bounds
                        ):
    """Генерация n-го количества портфелей"""
    # Инициализация записи весов активов и массива,
    # в который будут записываться данные по сформированным портфелям.
    results = np.zeros((3, num_portfolios))
    weights_record = []

    # Инициализация переменных, в которых будут храниться волатильность,
    # доходность и веса максимального кондиционного и модиф. коэф. Шарпа,
    # а также портфеля с мин. волатильностью (с огр. на мин. доходность)
    max_csr = -np.inf
    best_csr_ret = None
    best_csr_vol = None
    best_csr_weights = None

    max_msr = -np.inf
    best_msr_ret = None
    best_msr_vol = None
    best_msr_weights = None

    mv_vol = np.inf
    mv_ret = None
    mv_weights = None
    mv_sharpe = None

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

        portf_returns = daily_returns.dot(weights)
        portf_mean = portf_returns.mean()
        portf_std = portf_returns.std()
        ret = np.dot(weights, expected_return)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - rf) / vol

        results[0, i] = vol
        results[1, i] = ret
        results[2, i] = sharpe

        # Портфель с минимальной волатильностью и ограничением на мин. доход.
        if vol <= mv_vol and ret >= min_vol_return:
            mv_ret, mv_vol, mv_sharpe, mv_weights = ret, vol, sharpe, weights

        # Макс. CSR
        max_csr, best_csr_ret, best_csr_vol, best_csr_weights = find_max_csr(
            ret, vol, weights, portf_returns, portf_mean, daily_rf,
            max_csr, best_csr_ret, best_csr_vol, best_csr_weights, ALPHA
            )

        # Модифицированнный коэф. Шарпа
        max_msr, best_msr_ret, best_msr_vol, best_msr_weights = find_max_modified_sharpe(
            z, portf_returns, portf_mean, daily_rf, portf_std,
            ret, vol, weights, max_msr, best_msr_ret,
            best_msr_vol, best_msr_weights
            )

    inf_min_vol_portf = {'ret': mv_ret, 'vol': mv_vol,
                         'sharpe': mv_sharpe, 'weights': mv_weights
                         }
    inf_max_csr = {'max_csr': max_csr, 'ret': best_csr_ret,
                   'vol': best_csr_vol,
                   'weights': best_csr_weights
                   }
    inf_max_msr = {'max_msr': max_msr, 'ret': best_msr_ret,
                   'vol': best_msr_vol,
                   'weights': best_msr_weights
                   }

    return results, weights_record, inf_max_csr, inf_max_msr, inf_min_vol_portf


if __name__ == '__main__':
    # Константы
    NUM_PORTFOLIOS = 70000
    RISK_FREE_RATE = 0.0469
    ALPHA = 0.01
    MIN_VOLATILITY_RETURN = 0.4448

    # Параметры, связанные с константами
    daily_rf = RISK_FREE_RATE / 252
    z = norm.ppf(1 - ALPHA)

    tickers = [
        'AAPL', 'ASML', 'CME', 'DXCM', 'EQIX', 'IDXX', 'IRDM', 'IRM',
        'AVGO', 'NVDL', 'QQQ'
        ]

    # Ограничения на веса активов
    bounds = [(0.015, 0.21)] * len(tickers)

    data_2023 = get_data(tickers, '2023-01-01', '2023-12-31')

    expected_annual_return_2023, cov_matrix_2023, daily_returns_2023 = get_returns_and_cov_matrix(data_2023)

    # Построение Эффективной границы
    ef = EfficientFrontier(expected_annual_return_2023, cov_matrix_2023, weight_bounds=bounds)
    fig, ax = plt.subplots(figsize=(10, 7))
    plotting.plot_efficient_frontier(ef, ax=ax,
                                     show_assets=False)

    # Генерация 'облака результатов', CSR, MSR и портфеля с мин. вол.
    results, weights_record, inf_max_csr, inf_max_msr, inf_min_vol_portf = generate_portfolios(
        NUM_PORTFOLIOS, tickers, expected_annual_return_2023, cov_matrix_2023,
        RISK_FREE_RATE, daily_rf, ALPHA, MIN_VOLATILITY_RETURN,
        daily_returns_2023, bounds
        )

    # Портфель с максимальным коэффициентом Шарпа
    ef = EfficientFrontier(
        expected_annual_return_2023, cov_matrix_2023, weight_bounds=bounds
        )
    ef.max_sharpe()
    max_sharpe_ret, max_sharpe_vol, max_sharpe_sharpe = ef.portfolio_performance()

    # Портфель с максимальной доходностью и ограничением на риск
    ef_max_rt = EfficientFrontier(
        expected_annual_return_2023, cov_matrix_2023, weight_bounds=bounds
        )
    ef_max_rt.efficient_risk(target_volatility=0.1727)
    max_rt_ret, max_rt_vol, max_rt_sharpe = ef_max_rt.portfolio_performance()

    # Равновзвешенный портфель
    eq_weights = np.array([1/len(tickers)] * len(tickers))
    eq_ret = np.dot(eq_weights, expected_annual_return_2023)
    eq_vol = np.sqrt(np.dot(eq_weights.T, np.dot(cov_matrix_2023, eq_weights)))
    eq_sharp = (eq_ret - RISK_FREE_RATE) / eq_vol

    # Отображение точки выбранного портфеля (данные за 2024 г)
    data_2024 = get_data(tickers, '2024-01-01', '2024-12-31')
    expected_annual_return_2024, cov_matrix_2024, daily_returns_2024 = get_returns_and_cov_matrix(data_2024)
    portf_2024_weights = np.array(list(ef.clean_weights().values()))
    portf_2024_ret = np.dot(portf_2024_weights, expected_annual_return_2024)
    portf_2024_vol = np.sqrt(np.dot(
        portf_2024_weights.T, np.dot(cov_matrix_2024, portf_2024_weights)))

    # Вывод результатов в терминале
    print(f'\n📊 Максимальный Кондиционный коэф. Шарпа: {inf_max_csr['max_csr']:.4f}')
    print('Веса этого портфеля:')
    for ticker, weight in zip(tickers, inf_max_csr['weights']):
        print(f'{ticker}: {weight:.2%}')

    print(f'\n📊 Максимальный Модифицированный коэф. Шарпа: {inf_max_msr['max_msr']:.4f}')
    print('Веса этого портфеля:')
    for ticker, weight in zip(tickers, inf_max_msr['weights']):
        print(f'{ticker}: {weight:.2%}')

    print(f'\n📊 Максимальный коэффиент Шарпа: {max_sharpe_sharpe:.4f}')
    print('Веса этого портфеля:')
    for ticker, weight in zip(tickers, ef.clean_weights().values()):
        print(f'{ticker}: {weight:.2%}')

    print('\n📊 Минимальная волатильность (с огр. на доходность):')
    print(f'Sharpe: {inf_min_vol_portf['sharpe']:.4f}, Доходность: {inf_min_vol_portf['ret']:.4f}, Риск: {inf_min_vol_portf['vol']:.4f}')
    print('Веса этого портфеля:')
    for ticker, weight in zip(tickers, inf_min_vol_portf['weights']):
        print(f"{ticker}: {weight:.2%}")

    print('\n📊 Максимальная доходность (с огр. на риск):')
    print(f'Sharpe: {max_rt_sharpe:.4f}, Доходность: {max_rt_ret:.4f}, Риск: {max_rt_vol:.4f}')
    print('Веса этого портфеля:')
    for ticker, weight in zip(tickers, ef_max_rt.clean_weights().values()):
        print(f'{ticker}: {weight:.2%}')

    print('\n📊 Равновзвешенный портфель:')
    print(f'Sharpe: {eq_sharp:.4f}, Доходность: {eq_ret:.4f}, Риск: {eq_vol:.4f}')
    print('Веса равные для всех активов:')
    for ticker, weight in zip(tickers, eq_weights):
        print(f'{ticker}: {weight:.2%}')

    # Визуализация
    scatter = ax.scatter(
        results[0, :], results[1, :], c=results[2, :], cmap='viridis', alpha=0.4
        )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Коэффициент Шарпа')

    ax.scatter(max_sharpe_vol, max_sharpe_ret, marker='*', color='r', s=200,
               label='Макс. коэф. Шарп (выбранный портфель 2023 г.)', alpha=1
               )
    ax.scatter(inf_min_vol_portf['vol'], inf_min_vol_portf['ret'],
               marker='*', color='b', s=200, label='Мин. волат.', alpha=0.7
               )
    ax.scatter(
        eq_vol, eq_ret, marker='*', color='black', s=200,
        label='Равновзвешенный', alpha=0.7
        )
    ax.scatter(
        inf_max_csr['vol'], inf_max_csr['ret'],
        marker='*', color='grey', s=200, label='Макс. Кондиц. коэф. Шарпа',
        alpha=0.7
        )
    ax.scatter(
        inf_max_msr['vol'], inf_max_msr['ret'],
        marker='*', color='orange', s=200, label='Макс. Модиф. коэф. Шарпа',
        alpha=0.7
        )
    ax.scatter(
        max_rt_vol, max_rt_ret, marker='*', color='green', s=200,
        label='Макс. доходность', alpha=0.7
        )
    ax.scatter(
        portf_2024_vol, portf_2024_ret, marker='*', color='magenta', s=180,
        label='Выбранный портфель (2024 г.)', alpha=1
        )

    ax.set_title('Граница эффективности')
    ax.set_xlabel('Риск (волатильность)')
    ax.set_ylabel('Доходность')
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('efficient_frontier_nasdaq.png', dpi=300, bbox_inches='tight')
    plt.show()
