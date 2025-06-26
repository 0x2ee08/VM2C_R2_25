#!/usr/bin/env python3
"""
VN Weekly Analysis — Extended Back-test & Reporting
==================================================

What this script adds compared with the original
------------------------------------------------
1. **Full 2024 weekly simulation** – portfolio value, weekly P&L, realised risk
   (rolling annualised σ) recorded after every week.
2. **Dynamic re-balancing option** – if `DYNAMIC_REBALANCE = True` we solve a
   fresh Markowitz problem each Friday using data **only up to that week** and
   apply the new weights for the next week – exactly the requirement in section
   3(d) of the brief.
3. **Comprehensive outputs**
   •  A CSV log `weekly_log.csv` with one row per week of 2024, containing the
      items (a)–(d).
   •  A nicely formatted *summary* table printed to console at the end of the
      run (section 4).
   •  An interactive Matplotlib chart of cumulative capital, weekly P&L and
      rolling risk (section 5 – illustrations).  The chart is saved to
      `portfolio_2024.png` as well.
4. **Sharpe ratio** – computed using realised 2024 return and realised σ.

Configuration knobs live in the *Parameters* section near the top.
Everything else runs end-to-end in a single invocation:

    python vn_backtest_extended.py  [--no-dynamic]  [--plot-off]

Dependencies
------------
`pandas`, `numpy`, `matplotlib`  – standard stacks; `gurobipy` optional (only
needed when optimiser weights or dynamic re-balancing are requested).

Author: ChatGPT-o3, 2025-06-25
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Optional optimiser backend
try:
    import gurobipy as gp
    from gurobipy import GRB
except ModuleNotFoundError:
    gp = None  # graceful degradation – optimisation will be skipped

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Parameters — tweak here if you like
# ------------------------------------------------------------
FOLDER_1W: Path | str = "VN_1W"       # weekly bars (≥ 2021-12-31)
FOLDER_2024: Path | str = "VN_2024"   # weekly bars for 2024 (must include 2023-12-29 close)
SHARES_FILE: Path | str | None = "shares_input.csv"  # optional initial manual holdings
ANNUAL_FACTOR = 52                    # 1 week = 1/52 year
RISK_TARGET = 0.20                    # 20 % annual volatility cap
ALLOW_SHORT = False                   # allow short sales in optimiser
CAPITAL_START = 25_500_000_000.0      # VND 25.5 billion at 2023-12-29
DYNAMIC_REBALANCE = True              # set False for static buy-and-hold
ROLLING_WINDOW = None                 # None ⇒ use *all* history up to week-i
RF_RATE_ANNUAL = 0.0                  # risk-free (for Sharpe). Change if desired
PLOT_RESULTS = True                   # toggle via CLI

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------

def solve_markowitz(mu: pd.Series, cov: pd.DataFrame, sigma_max: float,
                     allow_short: bool = False) -> tuple[np.ndarray, float, float]:
    """Long-only (or unconstrained) Markowitz max-return subject to vol ≤ sigma_max."""

    if gp is None:
        raise RuntimeError("Gurobi not available – optimisation disabled.")

    symbols = mu.index.tolist()
    mu_vec = mu.values
    n = len(symbols)

    m = gp.Model("markowitz")
    w = m.addVars(n, lb=(None if allow_short else 0.0), name="w")
    m.setObjective(-gp.quicksum(mu_vec[i] * w[i] for i in range(n)), GRB.MINIMIZE)
    m.addConstr(gp.quicksum(w[i] for i in range(n)) == 1, "budget")

    quad = gp.QuadExpr()
    for i in range(n):
        for j in range(n):
            if cov.iat[i, j] != 0.0:
                quad.add(cov.iat[i, j] * w[i] * w[j])
    m.addQConstr(quad <= sigma_max ** 2, "risk")

    m.Params.OutputFlag = 0
    m.optimize()

    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Optimiser status {m.Status}")

    weights = np.array([w[i].X for i in range(n)])
    exp_ret = float(mu_vec @ weights)
    risk = float(np.sqrt(weights @ cov.values @ weights))
    return weights, exp_ret, risk


def annualise_weekly(series: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return series * ANNUAL_FACTOR


def annualise_sigma_weekly(series: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return series * math.sqrt(ANNUAL_FACTOR)


# ------------------------------------------------------------
# 0) CLI
# ------------------------------------------------------------
parser = argparse.ArgumentParser(description="VN Index weekly back-test - extended")
parser.add_argument("--no-dynamic", action="store_true", help="Turn OFF dynamic re-balancing")
parser.add_argument("--plot-off", action="store_true", help="Disable result plotting")
args = parser.parse_args()

if args.no_dynamic:
    DYNAMIC_REBALANCE = False
if args.plot_off:
    PLOT_RESULTS = False

# ------------------------------------------------------------
# 1) Load 2021-2023 weekly data – history for stats & t0 reference prices
# ------------------------------------------------------------

folder_hist = Path(FOLDER_1W)
if not folder_hist.is_dir():
    raise SystemExit(f"Historical folder not found: {folder_hist.resolve()}")

returns_dict: dict[str, pd.Series] = {}
price_at_2023: dict[str, float] = {}
all_dates_hist: pd.DatetimeIndex | None = None

for csv_file in glob.glob(str(folder_hist / "*.csv")):
    df = pd.read_csv(csv_file)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").set_index("datetime")

    # derive symbol
    symbol = df["symbol"].iloc[0] if "symbol" in df.columns and pd.notna(df["symbol"].iloc[0]) else Path(csv_file).stem

    # weekly pct return in percent units
    weekly_pct = df["close"].pct_change().mul(100).dropna()
    returns_dict[symbol] = weekly_pct

    # closing price at end of 2023 (assumes last Friday 2023-12-29)
    df_2023 = df.loc[df.index.year == 2023]
    if not df_2023.empty:
        price_at_2023[symbol] = float(df_2023["close"].iloc[-1])

    all_dates_hist = weekly_pct.index if all_dates_hist is None else all_dates_hist.union(weekly_pct.index)

if not returns_dict:
    raise SystemExit("No weekly history loaded – aborting.")

returns_hist_df = pd.DataFrame(index=sorted(all_dates_hist))
for sym, ser in returns_dict.items():
    returns_hist_df[sym] = ser.reindex(returns_hist_df.index)
returns_hist_df.dropna(how="all", inplace=True)

# ------------------------------------------------------------
# 2) Compute µ̂ and σ̂ from history (annualised, %) – section 2
# ------------------------------------------------------------

mu_weekly_pct = returns_hist_df.mean()
sigma_weekly_pct = returns_hist_df.std(ddof=0)

mu_annual_pct = annualise_weekly(mu_weekly_pct)
sigma_annual_pct = annualise_sigma_weekly(sigma_weekly_pct)

stats_summary = pd.DataFrame({
    "mu_annual_%": mu_annual_pct,
    "sigma_annual_%": sigma_annual_pct,
}).sort_index()

# ------------------------------------------------------------
# 3) Build correlation (ρ) and covariance (Σ_annual in decimal units)
# ------------------------------------------------------------

rho_weekly = returns_hist_df.corr()
returns_hist_dec = returns_hist_df.div(100)  # % -> decimal
Sigma_annual = returns_hist_dec.cov(ddof=0) * ANNUAL_FACTOR

symbols = stats_summary.index.tolist()

# ------------------------------------------------------------
# 4) Initial portfolio – optimiser or manual holdings
# ------------------------------------------------------------

if SHARES_FILE and Path(SHARES_FILE).is_file():
    df_sh = pd.read_csv(SHARES_FILE)
    if {"symbol", "shares"}.issubset(df_sh.columns):
        manual_shares = {str(r.symbol): float(r.shares) for r in df_sh.itertuples(index=False)}
    else:
        warnings.warn("Shares CSV missing columns; ignoring manual holdings.")
        manual_shares = {}
else:
    manual_shares = {}

if manual_shares:
    shares0 = {s: manual_shares.get(s, 0.0) for s in symbols}
    holding_mode = "Manual"
    weights0 = np.array([shares0[s] * price_at_2023[s] for s in symbols]) / CAPITAL_START
else:
    # derive optimiser weights w* (static, or used as week-0 weights if dynamic)
    try:
        w_opt, ret_opt, sig_opt = solve_markowitz(mu=mu_annual_pct/100, cov=Sigma_annual, sigma_max=RISK_TARGET, allow_short=ALLOW_SHORT)
    except Exception as exc:
        warnings.warn(f"Optimiser failed ({exc}); falling back to equal weights.")
        w_opt = np.full(len(symbols), 1/len(symbols))
        ret_opt = sig_opt = float("nan")

    shares0 = {s: (CAPITAL_START * w // price_at_2023[s]) for s, w in zip(symbols, w_opt)}
    holding_mode = "Optimiser"
    weights0 = w_opt

# ------------------------------------------------------------
# 5) Load 2024 weekly closing prices (must align with Friday week-end dates)
# ------------------------------------------------------------
folder_2024 = Path(FOLDER_2024)
if not folder_2024.is_dir():
    raise SystemExit(f"2024 data folder not found: {folder_2024.resolve()}")

prices_2024: dict[str, pd.Series] = {}
all_dates_2024: pd.DatetimeIndex | None = None

for sym in symbols:
    csv_file = folder_2024 / f"{sym}.csv"
    if not csv_file.is_file():
        warnings.warn(f"Missing 2024 data for {sym}; symbol excluded from back-test.")
        continue
    df = pd.read_csv(csv_file)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.sort_values("datetime").set_index("datetime")
    df_2024 = df.loc[df.index.year == 2024]
    if df_2024.empty:
        warnings.warn(f"{sym}: no 2024 rows – excluded.")
        continue
    prices_2024[sym] = df_2024["close"]
    all_dates_2024 = df_2024.index if all_dates_2024 is None else all_dates_2024.union(df_2024.index)

if not prices_2024:
    raise SystemExit("No 2024 price data loaded — cannot run back-test.")

dates_2024 = sorted(all_dates_2024)

# Align price series across symbols (forward-fill gaps inside the year)
prices_2024_df = pd.DataFrame(index=dates_2024)
for sym, ser in prices_2024.items():
    prices_2024_df[sym] = ser.reindex(prices_2024_df.index).fillna(method="ffill")

# Ensure starting date corresponds to end-2023 close for return calculation
start_date = pd.Timestamp("2023-12-29")
if start_date not in returns_hist_df.index:
    prices_0 = {s: price_at_2023[s] for s in symbols}
else:
    prices_0 = {s: price_at_2023[s] for s in symbols}

# ------------------------------------------------------------
# 6) Weekly simulation through 2024
# ------------------------------------------------------------
portfolio_values = []        # capital *after* close of each week
a_weekly_pnl = []            # profit this week (absolute VND)
rolling_sigma = []           # realised σ from week-1 .. i (annualised)
allocations_log = []         # dict of weights for each week (used for week i)

shares_hold = shares0.copy()
current_cap = CAPITAL_START
weekly_returns_portfolio = []

for i, date in enumerate(prices_2024_df.index, start=1):
    prices_week = prices_2024_df.loc[date]

    # ---- End-of-week portfolio valuation ----
    value_week = float(sum(shares_hold[s] * prices_week[s] for s in shares_hold))
    portfolio_values.append(value_week)

    # ---- Weekly P&L ----
    pnl_week = value_week - current_cap
    a_weekly_pnl.append(pnl_week)
    r_week = pnl_week / current_cap if current_cap else 0.0
    weekly_returns_portfolio.append(r_week)

    # ---- Rolling realised σ (annualised) ----
    sigma_realised = (np.std(weekly_returns_portfolio, ddof=0)
                      * math.sqrt(ANNUAL_FACTOR) if len(weekly_returns_portfolio) > 1 else 0.0)
    rolling_sigma.append(sigma_realised)

    # ---- Log current allocation (weights) ----
    weights_curr = {s: shares_hold[s] * prices_week[s] / value_week for s in shares_hold}
    allocations_log.append(weights_curr)

    # ---- Dynamic re-balance for next week (if enabled) ----
    if DYNAMIC_REBALANCE and gp is not None and i < len(prices_2024_df.index):
        # Build return history up to current week (inclusive)
        end_idx = returns_hist_df.index.get_loc(date, method="pad", tolerance=None) if date in returns_hist_df.index else None
        if end_idx is None:
            # If date beyond hist index, append weekly returns for 2024 up to date
            hist_ext = returns_hist_df.copy()
            returns_2024_to_date = prices_2024_df.iloc[:i].pct_change().mul(100)
            hist_ext = pd.concat([hist_ext, returns_2024_to_date])
        else:
            hist_ext = returns_hist_df.iloc[:end_idx+1]

        if ROLLING_WINDOW is not None and len(hist_ext) > ROLLING_WINDOW:
            hist_ext = hist_ext.iloc[-ROLLING_WINDOW:]

        mu_w = hist_ext.mean() / 100  # decimal
        cov_w = hist_ext.div(100).cov() * ANNUAL_FACTOR

        try:
            w_new, _, _ = solve_markowitz(mu_w, cov_w, sigma_max=RISK_TARGET)
        except Exception as exc:
            warnings.warn(f"Week {i}: optimiser failed ({exc}); keeping weights.")
            w_new = np.array(list(weights_curr.values()))

        # Convert weights to share counts for next week using *current* prices
        target_values = w_new * value_week
        for j, sym in enumerate(shares_hold):
            shares_hold[sym] = target_values[j] // prices_week[sym]
    # ---- if no rebalance, shares_hold stays ----

    current_cap = value_week  # reference for next week's P&L

# ------------------------------------------------------------
# 7) Post-simulation metrics (section 4)
# ------------------------------------------------------------
portfolio_values = np.array(portfolio_values)
weekly_returns_portfolio = np.array(weekly_returns_portfolio)
anual_return_realised = (portfolio_values[-1] / portfolio_values[0]) - 1
sigma_realised_annual = np.std(weekly_returns_portfolio, ddof=0) * math.sqrt(ANNUAL_FACTOR)
sharpe_ratio = ((anual_return_realised - RF_RATE_ANNUAL) / sigma_realised_annual) if sigma_realised_annual else float("nan")

# ------------------------------------------------------------
# 8) Outputs: CSV log, console summary, plot (optional)
# ------------------------------------------------------------
log_df = pd.DataFrame({
    "value": portfolio_values,
    "pnl": a_weekly_pnl,
    "rolling_sigma": rolling_sigma,
    "return_week": weekly_returns_portfolio,
}, index=prices_2024_df.index)
log_df.index.name = "week_end"
log_df.to_csv("weekly_log.csv", float_format="%.6f")

print("\n========== 2024 Weekly Back-test Summary ==========")
print(f"Initial capital            : {portfolio_values[0]:,.0f} VND")
print(f"Final capital              : {portfolio_values[-1]:,.0f} VND")
print(f"Realised return 2024       : {anual_return_realised:6.2%}")
print(f"Realised annual σ          : {sigma_realised_annual:6.2%}")
print(f"Sharpe ratio (rf={RF_RATE_ANNUAL:.2%}): {sharpe_ratio:6.2f}")
print("\n(See weekly_log.csv for full weekly details)")

if PLOT_RESULTS:
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(log_df.index, log_df["value"], label="Portfolio value")
    ax1.set_ylabel("Capital (VND)")
    ax1.set_title("Portfolio value, weekly P&L and rolling risk – 2024")
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.bar(log_df.index, log_df["pnl"], alpha=0.2, label="Weekly P&L")
    ax2.plot(log_df.index, log_df["rolling_sigma"], linestyle="--", label="Rolling σ (annual)")
    ax2.set_ylabel("P&L / Risk")

    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lb = ax.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(lb)
    ax1.legend(lines, labels, loc="upper left")
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig("portfolio_2024.png", dpi=200)
    plt.show()

print("\n✅ Back-test complete. Files generated: weekly_log.csv, portfolio_2024.png")
