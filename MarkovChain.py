import numpy as np
import pandas as pd
from scipy.stats import t, norm

# ===============================
# 1. Black-Scholes Call
# ===============================

def black_scholes_call(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    from scipy.stats import norm
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ===============================
# 2. Fat-tailed price paths
# ===============================

def simulate_fat_tailed_paths(
    S0,
    mu=0.05,
    sigma=0.30,
    T=2.0,
    days_per_year=252,
    n_paths=5000,
    nu=3
):
    """
    Simulate geometric-like paths with Student-t shocks.
    """
    if nu <= 2:
        raise ValueError("nu must be > 2 for finite variance")

    steps = int(T * days_per_year)
    dt = 1.0 / days_per_year

    # Student-t shocks with unit-ish variance
    var_t = nu / (nu - 2)
    scale = np.sqrt(var_t)
    eps = t.rvs(df=nu, size=(n_paths, steps)) / scale

    log_S = np.zeros((n_paths, steps + 1))
    log_S[:, 0] = np.log(S0)

    drift = (mu - 0.5 * sigma**2) * dt
    vol_step = sigma * np.sqrt(dt)

    for i in range(steps):
        log_S[:, i+1] = log_S[:, i] + drift + vol_step * eps[:, i]

    return np.exp(log_S)  # shape: (n_paths, steps+1)


# ===============================
# 3. Regime detection
# ===============================

def detect_regimes(
    S_paths,
    days_per_year=252,
    vol_window=20,
    ma_window=50,
    low_vol_thresh=0.15,   # annualized vol
    high_vol_thresh=0.35   # annualized vol
):
    """
    Classify each (path, time) into regimes based on:
    - Realized vol (20d)
    - 50d moving average trend

    Regimes:
    0 = UNKNOWN (insufficient history)
    1 = CALM_TREND
    2 = NEUTRAL
    3 = STRESSED
    """
    n_paths, n_steps_plus_1 = S_paths.shape
    n_steps = n_steps_plus_1 - 1

    # log returns per day
    log_S = np.log(S_paths)
    log_returns = log_S[:, 1:] - log_S[:, :-1]  # shape: (n_paths, n_steps)

    # containers
    realized_vol = np.full((n_paths, n_steps), np.nan)
    ma_50 = np.full((n_paths, n_steps_plus_1), np.nan)

    # rolling realized vol (per path)
    for p in range(n_paths):
        for t in range(vol_window, n_steps + 1):
            window = log_returns[p, t-vol_window:t]
            realized_vol[p, t-1] = np.std(window) * np.sqrt(days_per_year)

    # rolling 50d moving average
    for p in range(n_paths):
        for t in range(ma_window, n_steps_plus_1):
            ma_50[p, t] = np.mean(S_paths[p, t-ma_window+1:t+1])

    # classify regimes
    regimes = np.zeros((n_paths, n_steps_plus_1), dtype=int)

    for p in range(n_paths):
        for t in range(n_steps_plus_1):
            # need both vol and MA history
            if t == 0 or t-1 < vol_window or t < ma_window:
                regimes[p, t] = 0  # UNKNOWN
                continue

            rv = realized_vol[p, t-1]
            ma = ma_50[p, t]
            if np.isnan(rv) or np.isnan(ma):
                regimes[p, t] = 0
                continue

            price = S_paths[p, t]

            # simple trend flag: price above MA and MA rising over last 5 days
            if t >= ma_window + 5:
                ma_prev = ma_50[p, t-5]
            else:
                ma_prev = ma_50[p, t]  # fallback
            ma_slope_up = (not np.isnan(ma_prev)) and (ma > ma_prev)

            # classify
            if rv < low_vol_thresh and price > ma and ma_slope_up:
                regimes[p, t] = 1  # CALM_TREND
            elif rv > high_vol_thresh:
                regimes[p, t] = 3  # STRESSED
            else:
                regimes[p, t] = 2  # NEUTRAL

    return regimes


REGIME_LABELS = {
    0: "UNKNOWN",
    1: "CALM_TREND",
    2: "NEUTRAL",
    3: "STRESSED",
}


# ===============================
# 4. PMCC Simulation (single ticker)
# ===============================

def simulate_pmcc_for_ticker(
    symbol,
    S0,
    r=0.02,
    sigma_iv=0.35,
    T_leaps=2.0,
    days_per_year=252,
    n_paths=5000,
    nu=3,
    leaps_ITM_mult=0.7,
    short_DTE_days=45,
    short_OTM_mult=1.05,
    regime_use=False
):
    """
    PMCC under fat-tailed dynamics.
    If regime_use=True, we won't write short calls in STRESSED regime
    (this is just an example rule).
    """
    # 1) simulate paths
    S_paths = simulate_fat_tailed_paths(
        S0=S0,
        mu=r,
        sigma=sigma_iv,
        T=T_leaps,
        days_per_year=days_per_year,
        n_paths=n_paths,
        nu=nu
    )
    n_steps = S_paths.shape[1] - 1
    dt = 1.0 / days_per_year

    # 2) regimes
    regimes = detect_regimes(S_paths, days_per_year=days_per_year)

    # 3) LEAPS setup
    K_leaps = S0 * leaps_ITM_mult
    leaps_premium = black_scholes_call(S0, K_leaps, T_leaps, r, sigma_iv)

    # 4) short call schedule
    short_interval = short_DTE_days
    short_expiry_indices = list(range(short_interval, n_steps + 1, short_interval))

    pmcc_pl = np.zeros(n_paths)

    for path_idx in range(n_paths):
        S_path = S_paths[path_idx, :]
        path_regimes = regimes[path_idx, :]

        t_index = 0
        short_cashflow = 0.0

        for expiry_idx in short_expiry_indices:
            T_short = (expiry_idx - t_index) * dt
            if T_short <= 0:
                continue

            # Optionally: only write short calls if regime at entry is not STRESSED
            if regime_use:
                reg_code = path_regimes[t_index]
                if reg_code == 3:  # STRESSED
                    t_index = expiry_idx
                    continue

            S_start = S_path[t_index]
            K_short = S_start * short_OTM_mult
            short_premium = black_scholes_call(S_start, K_short, T_short, r, sigma_iv)
            short_cashflow += short_premium

            S_expiry = S_path[expiry_idx]
            payoff_short = -max(S_expiry - K_short, 0.0)
            short_cashflow += payoff_short

            t_index = expiry_idx

        # LEAPS payoff at final maturity
        S_T = S_path[-1]
        leaps_payoff = max(S_T - K_leaps, 0.0)

        pmcc_pl[path_idx] = leaps_payoff - leaps_premium + short_cashflow

    return {
        "symbol": symbol,
        "S_paths": S_paths,
        "regimes": regimes,
        "pmcc_pl": pmcc_pl,
        "K_leaps": K_leaps,
        "leaps_premium": leaps_premium,
    }


# ===============================
# 5. Summary helper
# ===============================

def summarize_pnl(pnl, label="Strategy"):
    print(f"\n=== {label} ===")
    print(f"Mean P&L:       {np.mean(pnl):.2f}")
    print(f"Median P&L:     {np.median(pnl):.2f}")
    print(f"Std Dev:        {np.std(pnl):.2f}")
    print(f"5th percentile: {np.percentile(pnl, 5):.2f}")
    print(f"1st percentile: {np.percentile(pnl, 1):.2f}")
    print(f"Min:            {np.min(pnl):.2f}")
    print(f"Max:            {np.max(pnl):.2f}")


# ===============================
# 6. Example run (META-like)
# ===============================

if __name__ == "__main__":
    np.random.seed(42)

    S0_META = 500.0
    sigma_META = 0.35

    res_naive = simulate_pmcc_for_ticker(
        "META",
        S0_META,
        sigma_iv=sigma_META,
        n_paths=5000,
        nu=3,
        regime_use=False  # write short calls all the time
    )

    res_regime = simulate_pmcc_for_ticker(
        "META",
        S0_META,
        sigma_iv=sigma_META,
        n_paths=5000,
        nu=3,
        regime_use=True   # skip short calls in STRESSED regimes
    )

    summarize_pnl(res_naive["pmcc_pl"], "META PMCC (always short)")
    summarize_pnl(res_regime["pmcc_pl"], "META PMCC (regime-aware)")

    # Example: look at how often each regime occurs at final time
    final_regimes = res_regime["regimes"][:, -1]
    unique, counts = np.unique(final_regimes, return_counts=True)
    print("\nFinal-time regimes frequency:")
    for u, c in zip(unique, counts):
        print(REGIME_LABELS[u], ":", c)
