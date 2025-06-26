from __future__ import annotations

import glob
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

FOLDER_1W: Path | str = "VN_1W"         # weekly bars, 2021‑2023
FOLDER_2024: Path | str = "VN_2024"     # 2024 data (any frequency)
SHARES_FILE: Path | str | None = "share_t2_us.csv"  # optional manual holdings
ANNUAL_FACTOR: int = 52                  # 1 week = 1/52 year
RISK_TARGET: float = 0.20               # 20 % annual volatility cap
ALLOW_SHORT: bool = False                # allow short sales in optimiser
CAPITAL_START: float = 25_500_000_000.0  # VND 25 billion at end‑2023

folder_path = Path(FOLDER_1W)
if not folder_path.is_dir():
    raise SystemExit(f"Folder not found: {folder_path.resolve()}")
    
returns_dict: dict[str, pd.Series] = {}
init_price: dict[str, float] = {}
hatcaicc: list[dict[str, float]] = [{} for _ in range(55)]
all_dates: pd.DatetimeIndex | None = None
all_dates_24: pd.DatetimeIndex | None = None
returns_dict_24: dict[str, pd.Series] = {}

for csv_file in glob.glob(str(folder_path / "*.csv")):
    try:
        df = pd.read_csv(csv_file)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime").set_index("datetime")

        if df.index.hasnans:
            warnings.warn(f"Missing datetime values in {csv_file}; affected rows dropped.")
            df = df[~df.index.isna()]

        symbol = (
            df["symbol"].iloc[0]
            if "symbol" in df.columns and pd.notna(df["symbol"].iloc[0])
            else Path(csv_file).stem
        )

        # Weekly % return – multiply by 100 so the series is in percent units
        weekly_pct = df["close"].pct_change().mul(100).dropna()
        returns_dict[symbol] = weekly_pct

        # Closing price at the end of 2023 (last trading date in 2023)
        df_2023 = df.loc[df.index.year == 2023]
        if not df_2023.empty:
            init_price[symbol] = float(df_2023["close"].iloc[-1])

        all_dates = (
            weekly_pct.index
            if all_dates is None
            else all_dates.union(weekly_pct.index)
        )

    except Exception as exc:
        warnings.warn(f"⚠️  Skipping {csv_file}: {exc}")

if not returns_dict:
    raise SystemExit("No valid CSV files were processed.")

# Align series on a common calendar so matrix maths works nicely
returns_df = pd.DataFrame(index=sorted(all_dates))
for sym, ser in returns_dict.items():
    returns_df[sym] = ser.reindex(returns_df.index)

returns_df.dropna(how="all", inplace=True)
returns_df = returns_df.sort_index(axis=1)

mu_weekly_pct = returns_df.mean(skipna=True)
mu_annual_pct = mu_weekly_pct * ANNUAL_FACTOR

summary = pd.DataFrame({
    "mu_annual_%": mu_annual_pct,
}).sort_index()

# Convert weekly % returns → decimal, then annualise covariance (var ∝ t)
returns_dec_weekly = returns_df.div(100)            # % → decimal
Σ_annual = returns_dec_weekly.cov(ddof=0) * ANNUAL_FACTOR
print(Σ_annual)

try:
    import gurobipy as gp
    from gurobipy import GRB

    symbols = summary.index.to_list()
    μ_vec = summary["mu_annual_%"].values / 100              # decimal form

    # Gurobi setup
    m = gp.Model("Markowitz")
    w = m.addVars(len(symbols), lb=0.0 if not ALLOW_SHORT else -GRB.INFINITY, name="w")

    # Maximise expected return (Gurobi minimises by default)
    m.setObjective(-gp.quicksum(μ_vec[i] * w[i] for i in range(len(symbols))), GRB.MINIMIZE)

    # Budget constraint  ∑w = 1
    m.addConstr(gp.quicksum(w[i] for i in range(len(symbols))) == 1, "budget")

    # Risk constraint  wᵀ Σ w ≤ σ_max²
    quad = gp.QuadExpr()
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            if Σ_annual.iat[i, j] != 0.0:  # small speed‑up
                quad.add(Σ_annual.iat[i, j] * w[i] * w[j])
    m.addQConstr(quad <= RISK_TARGET ** 2, "risk")

    m.Params.OutputFlag = 0  # silence solver log
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        w_opt = np.array([w[i].X for i in range(len(symbols))])
        ret_opt = float(μ_vec @ w_opt)
        sig_opt = float(np.sqrt(w_opt @ Σ_annual.values @ w_opt))
    else:
        warnings.warn(f"Solver returned status {m.Status}; no optimal solution.")
        w_opt = ret_opt = sig_opt = None
except ModuleNotFoundError:
    warnings.warn("Gurobi not installed – optimisation step skipped.")
    symbols = w_opt = ret_opt = sig_opt = None
except Exception as exc:
    warnings.warn(f"Markowitz optimisation failed: {exc}")
    symbols = w_opt = ret_opt = sig_opt = None

manual_shares: dict[str, float] = {}
if SHARES_FILE is not None and Path(SHARES_FILE).is_file():
    df_sh = pd.read_csv(SHARES_FILE)
    if {"symbol", "shares"}.issubset(df_sh.columns):
        manual_shares = {str(r.symbol): float(r.shares) for r in df_sh.itertuples(index=False)}
    else:
        warnings.warn(f"{SHARES_FILE} missing required columns 'symbol' and 'shares'. Ignoring file.")
else:
    if SHARES_FILE is not None:
        warnings.warn(f"Manual shares file '{SHARES_FILE}' not found. Defaulting to optimiser weights.")

fin_price: dict[str, float] = {}
price = {}
if symbols is None:
    raise SystemExit("No symbols available for simulation – see earlier warnings.")

folder_2024 = Path(FOLDER_2024)
if not folder_2024.is_dir():
    warnings.warn(f"2024 folder not found: {folder_2024.resolve()}; profit simulation skipped.")
    fin_price = {}
else:
    for sym in symbols:
        csv_file = folder_2024 / f"{sym}.csv"
        if not csv_file.is_file():
            warnings.warn(f"Missing 2024 data for {sym}; profit simulation skipped for this symbol.")
            continue
        df = pd.read_csv(csv_file)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime").set_index("datetime")
        # df_2024 = df.loc[df.index.year == 2024]
        df_2024 = df
        for dm in range(len(df_2024["close"])):
            hatcaicc[dm][sym] = df_2024["close"].iloc[dm]
        if df_2024.empty:
            warnings.warn(f"No 2024 prices for {sym}; profit simulation skipped for this symbol.")
            continue
        fin_price[sym] = float(df_2024["close"].iloc[-1])
        price[sym] = df_2024["close"]
        weekly_pct_24 = df_2024["close"].pct_change().mul(100).dropna()
        returns_dict_24[sym] = weekly_pct_24
        all_dates_24 = (
            weekly_pct_24.index
            if all_dates_24 is None
            else all_dates_24.union(weekly_pct_24.index)
        )
# print(all_dates_24)
returns_df_24 = pd.DataFrame(index=sorted(all_dates_24))
for sym, ser in returns_dict_24.items():
    returns_df_24[sym] = ser.reindex(returns_df_24.index)

# Drop rows where every symbol is NaN (e.g. holiday weeks)

# print(returns_df)
# print(returns_df_24)
returns_df_24.dropna(how="all", inplace=True)
# print(returns_df_24)
# ------------------------------------------------------------
# 7)  Determine which holdings to use
# ------------------------------------------------------------
if manual_shares:
    # Use manual holdings supplied by user
    a_shares = {s: manual_shares.get(s, 0.0) for s in symbols}
    holding_mode = "Manual"
else:
    # No manual holdings – fall back to optimiser weights (if available)
    if w_opt is None:
        raise SystemExit("Neither manual shares nor optimiser weights available. Nothing to simulate.")
    a_shares = {s: CAPITAL_START * w // init_price[s] for s, w in zip(symbols, w_opt)}
    holding_mode = "Optimiser"


P_start = sum(a_shares[s] * init_price[s] for s in symbols)
w_actual = np.array([a_shares[s] * init_price[s] / P_start for s in symbols])
sigma_pct = 0
print(P_start)
P_prev = P_start
leftMoney = CAPITAL_START - P_start
for ii in range(1, 54):
    mu_weekly_pct_ewma = returns_df.ewm(span=100, adjust=False).mean().iloc[-1]
    mu_weekly_pct = returns_df.mean(skipna=True)
    mu_annual_pct = mu_weekly_pct_ewma * ANNUAL_FACTOR #Try no ewma?

    # returns_last_100 = returns_df.tail(100)
    # mu_weekly_pct = returns_last_100.mean(skipna=True)
    # mu_annual_pct = mu_weekly_pct * ANNUAL_FACTOR

    summary = pd.DataFrame({
        "mu_annual_%": mu_annual_pct,
    }).sort_index()
    symbols = summary.index.to_list()
    μ_vec = summary["mu_annual_%"].values / 100              # decimal form
    m = gp.Model("Markowitz")
    w = m.addVars(len(symbols), lb=0.0 if not ALLOW_SHORT else -GRB.INFINITY, name="w")
    m.setObjective(-gp.quicksum(μ_vec[i] * w[i] for i in range(len(symbols))), GRB.MINIMIZE)
    m.addConstr(gp.quicksum(w[i] for i in range(len(symbols))) == 1, "budget")
    quad = gp.QuadExpr()
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            if Σ_annual.iat[i, j] != 0.0:  # small speed‑up
                quad.add(Σ_annual.iat[i, j] * w[i] * w[j])
    m.addQConstr(quad <= RISK_TARGET ** 2, "risk")
    m.Params.OutputFlag = 0  # silence solver log
    m.optimize()
    w_opt = np.array([w[i].X for i in range(len(symbols))])
    ret_opt = float(μ_vec @ w_opt)
    sig_opt = float(np.sqrt(w_opt @ Σ_annual.values @ w_opt))

    # print(w_opt, ret_opt, sig_opt)
    CAPITAL_START = P_prev + leftMoney

    a_shares = {s: CAPITAL_START * w // init_price[s] for s, w in zip(symbols, w_opt)}
    holding_mode = "Optimiser"

    P_next = sum(a_shares[s] * price[s].iloc[ii] for s in symbols)
    
    print("Tuần ", ii)
    print("Giá trị danh mục đầu tư:", P_next)
    print("Lãi/Lỗ so với tuần trước:", (P_next - P_prev)/P_prev)
    print("Lãi/lỗ so với số tiền ban đầu:", (P_next - P_start)/P_start)
    P_prev = P_next
    new_index = len(returns_df)  # next available row index
    returns_df.loc[new_index] = returns_df_24.iloc[ii - 1]

    returns_dec_weekly = returns_df.div(100)            # % → decimal
    Σ_annual = returns_dec_weekly.cov(ddof=0) * ANNUAL_FACTOR
    sigma_actual = float(np.sqrt(w_actual @ Σ_annual.values @ w_actual))
    sigma_pct = sigma_actual
    print("Rủi ro:", sigma_actual)
    print("--------")
    init_price = hatcaicc[ii]
    # print(init_price)
    # break
P_start = 25_500_000_000
P_end   = P_prev
profit_vnd = P_prev - P_start
profit_pct = profit_vnd / P_start if P_start else float("nan")
print("Lợi nhuận thực tế (VND):", profit_vnd)
print("Lợi nhuận thực tế (%)", profit_pct)
print("Độ rủi ro vào cuối năm 2024:", sigma_pct)
print("Tỉ suất Sharpe:", profit_pct/sigma_pct)
# End of file
