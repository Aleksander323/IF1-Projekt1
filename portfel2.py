import csv
import numpy as np
import scipy.stats as si
from numpy.random import normal
import matplotlib.pyplot as plt
from scipy.stats import normaltest, shapiro, anderson

dane = []
with open('/Users/aleksanderzurowski/Downloads/Dane historyczne 2022.csv') as csvfile:
    rows = csv.reader(csvfile, delimiter=",")
    for row in rows:
        dane.append(row[1])

dane.remove('Ostatnio')
dane_hist = [float(slowo.replace('.', '').replace(',', '.')) for slowo in dane]
dane_hist.append(2266.92)
dane_hist.reverse()


def bs_price(S, K, T, r, sigma, type='call'):
    assert type in ['call', 'put']

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    if type == 'call':
        return call_price
    else:
        put_price = call_price + K * np.exp(-r * T) - S
        return put_price


def get_delta(S, K, T, r, sigma, type='call'):
    assert type in ['call', 'put']

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    delta = si.norm.cdf(d1)
    if type == 'call':
        return delta
    else:
        return delta - 1


def BM(n, T=1):
    Z = [normal(0, 1) for _ in range(n+1)]
    points = [0]*(n + 1)
    for i in range(1, n+1):
        points[i] = points[i-1] + np.sqrt(T/n)*Z[i]
    return points


def GBM(n, points, alpha, sigma, S0):
    S = [0]*(n + 1)
    S[0] = S0
    t = [i / n for i in range(n+1)]

    for i in range(1, n+1):
        S[i] = S0 * np.exp(alpha * t[i] + sigma * points[i])

    return S


def delta_hedge_simulation(S0, K, T, r, sigma, mu, dt=1 / 252):
    """Simulates a delta hedging strategy for a short call position."""
    # Generate GBM path
    n = int(1/dt)
    points = BM(n)
    alpha = mu - sigma**2/2
    S_t = GBM(n, points, alpha, sigma, S0)
    # S_t = dane_hist
    # Initial Black-Scholes price
    call_price = bs_price(S0, K, T, r, sigma)
    delta = get_delta(S0, K, T, r, sigma)

    # Short the call, buy delta shares
    cash_position = call_price - delta * S0  # Initial cash position
    stock_position = delta
    portfolio_value = call_price
    stan_portfela = [portfolio_value]

    # Iterate over time steps to rebalance hedge
    for i in range(1, len(S_t)-1):
        tau = T - i * dt  # Time remaining
        new_delta = get_delta(S_t[i], K, tau, r, sigma)
        cash_position = cash_position * np.exp(r * dt) - (new_delta - stock_position) * S_t[i]
        stock_position = new_delta
        portfolio_value = cash_position + stock_position * S_t[i]
        stan_portfela.append(portfolio_value)

    # Final profit/loss
    final_pnl = cash_position * np.exp(r * dt) + stock_position * S_t[-1] - max(S_t[-1] - K, 0)  # Payoff of the short call
    stan_portfela.append(final_pnl)
    return final_pnl, S_t, stan_portfela


def delta_hedge_simulation2(S0, K, T, r, sigma, mu, n_hedgepoints):
    """Simulates a delta hedging strategy for a short call position."""
    # Generate GBM path
    dt = T/n_hedgepoints
    points = BM(n_hedgepoints)
    alpha = mu - sigma**2/2
    S_t = GBM(n_hedgepoints, points, alpha, sigma, S0)
    # S_t = dane_hist
    # Initial Black-Scholes price
    call_price = bs_price(S0, K, T, r, sigma)
    delta = get_delta(S0, K, T, r, sigma)

    # Short the call, buy delta shares
    cash_position = call_price - delta * S0  # Initial cash position
    stock_position = delta
    portfolio_value = call_price
    stan_portfela = [portfolio_value]

    # Iterate over time steps to rebalance hedge
    for i in range(1, len(S_t)-1):
        tau = T - i * dt  # Time remaining
        new_delta = get_delta(S_t[i], K, tau, r, sigma)
        cash_position = cash_position * np.exp(r * dt) - (new_delta - stock_position) * S_t[i]
        stock_position = new_delta
        portfolio_value = cash_position + stock_position * S_t[i]
        stan_portfela.append(portfolio_value)

    # Final profit/loss
    final_pnl = cash_position * np.exp(r * dt) + stock_position * S_t[-1] - max(S_t[-1] - K, 0)  # Payoff of the short call
    stan_portfela.append(final_pnl)
    return final_pnl, S_t, stan_portfela


# S0, K, T, r, sigma, mu = 100, 100, 1, 0.05, 0.5, 0.15


def delta_hedging_sample(S0, K, T, r, sigma, mu, n_hedgepoints, n=500):
    pnl_results = []
    paths = []
    stany = []

    for _ in range(n):
        final_pnl, S_t, stan_portfela = delta_hedge_simulation2(S0, K, T, r, sigma, mu, n_hedgepoints)
        pnl_results.append(final_pnl)
        paths.append(S_t)
        stany.append(stan_portfela)

    return pnl_results, paths, stany


S0, K, T, r, sigma, mu, n_hedgepoints = 2266.92, 1800, 1, 0.0346, 0.2656, 0.0265, 252
#pnl_results, _, _ = delta_hedging_sample(S0, K, T, r, sigma, mu, n_hedgepoints)


def draw_histogram(pnl_results, bins=np.arange(-30, 30, 1)):
    std = np.std(pnl_results)
    plt.hist(pnl_results, bins=bins, alpha=0.7, color='blue', edgecolor='black', density=True)
    plt.axvline(np.mean(pnl_results), color='r', linestyle="--", label=f"Mean P/L: {np.mean(pnl_results):.2f}")
    plt.axvline(std, color='r', linestyle="--", label=f"Standard deviation P/L: {np.std(pnl_results):.2f}")
    plt.axvline(-std, color='r', linestyle="--")
    plt.legend()
    plt.show()


def draw_quantiles(pnl_results):
    quantiles = np.percentile(pnl_results, [x for x in range(5, 96, 5)])  # Obliczenie kwantyli
    xax = [x / 100 for x in range(5, 96, 5)]
    plt.plot(xax, quantiles, marker='o', linestyle='-')
    #plt.show()


def get_std(n_hedge, sigma):
    pnl_results, _, _ = delta_hedging_sample(S0, K, T, r, sigma, mu, n_hedge)

    return np.std(pnl_results)


vectorized_func = np.vectorize(get_std)
#hedge_points = np.array([1, 2, 5, 10, 15, 20, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
hedge_points = np.array([x for x in range(1, 21)])
sigmas = np.array([x/100 for x in range(1, 51, 3)])
results = vectorized_func(252, sigmas)
plt.plot(sigmas, results, marker='o', linestyle='-')
#draw_histogram(pnl_results)
#draw_quantiles(pnl_results)
plt.show()

#print(normaltest(pnl_results))
#print(shapiro(pnl_results))



