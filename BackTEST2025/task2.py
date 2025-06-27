#!/usr/bin/env python3
"""
VN Weekly Analysis
==================

A single-file workflow that
1.  Reads every *.csv* file inside the **VN_1W/** directory (assumed weekly bars).
2.  Computes weekly simple returns R_t  = (C_t − C_{t−1}) / C_{t−1} in **percent**.
3.  Derives per-symbol statistics
        •     μ̂  – expected annual return  ( % p.a. )
        •     σ̂  – annual volatility       ( % p.a. )
4.  Builds the weekly correlation matrix ρ and the **annual** covariance matrix Σ.
5.  Solves a long-only Markowitz problem – maximise E[r] subject to portfolio
   volatility ≤ σ_max.
6.  Simulates investing VND 25 billion at the end of 2023 according to the
   optimal weights and shows the portfolio value at the end of 2024, along with
   profit in both VND and % terms.

All printed outputs are rounded for readability; underlying maths is done at
full precision.
"""

from __future__ import annotations

import glob
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Parameters — tweak here if you like
# ------------------------------------------------------------
FOLDER_1W: Path | str = "US_1W"         # weekly bars, 2021-2023
FOLDER_2024: Path | str = "US_2024"     # 2024 data (any frequency)
ANNUAL_FACTOR: int = 52                  # 1 week = 1/52 year
RISK_TARGET: float = 0.25                # 20 % annual volatility cap
ALLOW_SHORT: bool = True    # set True if you want to allow short sales
CAPITAL_START: float = 1_000_000.0  # VND 25 billion at end-2023

# ------------------------------------------------------------
# 1)  Load weekly files and build a returns DataFrame
# ------------------------------------------------------------
folder_path = Path(FOLDER_1W)
if not folder_path.is_dir():
    raise SystemExit(f"Folder not found: {folder_path.resolve()}")

returns_dict: dict[str, pd.Series] = {}
init_price: dict[str, float] = {}
all_dates: pd.DatetimeIndex | None = None

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

# Drop rows where every symbol is NaN (e.g. holiday weeks)
returns_df.dropna(how="all", inplace=True)

returns_df = returns_df.sort_index(axis=1)
# ------------------------------------------------------------
# 2)  Per-symbol μ and σ (all still in percent here)
# ------------------------------------------------------------
mu_weekly_pct = returns_df.mean(skipna=True)
σ_weekly_pct = returns_df.std(ddof=0, skipna=True)  # population stdev

mu_annual_pct = mu_weekly_pct * ANNUAL_FACTOR
σ_annual_pct = σ_weekly_pct * math.sqrt(ANNUAL_FACTOR)

summary = pd.DataFrame({
    "mu_annual_%": mu_annual_pct,
    "sigma_annual_%": σ_annual_pct,
}).sort_index()

# Convert weekly % returns → decimal, then annualise covariance (var ∝ t)
print(returns_df)
returns_dec_weekly = returns_df.div(100)            # % → decimal
Σ_annual = returns_dec_weekly.cov(ddof=0) * ANNUAL_FACTOR
# Σ_annual = Σ_annual.sort_index(axis=0).sort_index(axis=1)
print(Σ_annual)
# ------------------------------------------------------------
# 4)  Markowitz optimisation (long-only unless ALLOW_SHORT=True)
# ------------------------------------------------------------
try:
    import gurobipy as gp
    from gurobipy import GRB

    symbols = summary.index.to_list()
    μ_vec = summary["mu_annual_%"].values / 100              # decimal form
    print("μ_vec")
    print(μ_vec)
    print("Σ_annual")
    print(Σ_annual)

    m = gp.Model("portfolio_optimization")

    # Biến quyết định
    n = len(symbols)
    w = m.addVars(n, lb=-1, ub=1, name="w")
    u = m.addVars(n, lb=0, name="u")            # |w_i|
    neg_w = m.addVars(n, lb=0, name="neg_w")    # phần âm của w_i
    is_neg = m.addVars(n, vtype=GRB.BINARY, name="is_neg")  # w_i < 0

    M = 1e2  # hệ số lớn dùng trong ràng buộc logic

    # Mục tiêu
    m.setObjective(gp.quicksum(w[i] * μ_vec[i] for i in range(n)), GRB.MAXIMIZE)

    # Ràng buộc phương sai
    m.addQConstr(
        gp.QuadExpr(sum(w[i] * Σ_annual.iat[i, j] * w[j] for i in range(n) for j in range(n))) <= RISK_TARGET*RISK_TARGET,
        name="variance"
    )

    # Ràng buộc ||w||_1 = 1
    for i in range(n):
        m.addConstr(u[i] >= w[i])
        m.addConstr(u[i] >= -w[i])
    m.addConstr(gp.quicksum(u[i] for i in range(n)) == 1, name="l1_norm")

    # Ràng buộc phần âm của w
    for i in range(n):
        m.addConstr(neg_w[i] >= -w[i])
        m.addConstr(neg_w[i] <= M * is_neg[i])  # neg_w > 0 chỉ khi is_neg = 1
        m.addConstr(w[i] <= 0 + M * (1 - is_neg[i]))  # w <= 0 nếu is_neg = 1
    m.addConstr(gp.quicksum(neg_w[i] for i in range(n)) <= 0.3, name="negative_sum_limit")

    # Tối ưu hóa
    m.optimize()
    # # Gurobi setup
    # m = gp.Model("Markowitz")
    # w = m.addVars(len(symbols), lb=0.0, ub=1.0, name="w")

    # # Mục tiêu: tối đa hóa lợi suất kỳ vọng danh mục
    # m.setObjective(gp.quicksum(w[i] * μ_vec[i] for i in range(len(symbols))), GRB.MAXIMIZE)

    # # Ràng buộc 1: tổng trọng số bằng 1 (toàn bộ vốn đầu tư)
    # m.addConstr(gp.quicksum(w[i] for i in range(len(symbols))) == 1, name="Budget")

    # # Ràng buộc 2: độ lệch chuẩn danh mục không vượt quá 20%
    # # Tức là phương sai danh mục <= 0.2^2
    # portfolio_variance = gp.QuadExpr()
    # for i in range(len(symbols)):
    #     for j in range(len(symbols)):
    #         portfolio_variance += w[i] * w[j] * Σ_annual.iat[i, j]
    # m.addQConstr(portfolio_variance <= 0.2 ** 2, name="RiskBound")

    # m.optimize()

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

# ------------------------------------------------------------
# 5)  Profit simulation 2023-12-31 → 2024-12-31
# ------------------------------------------------------------
fin_price: dict[str, float] = {}
if w_opt is not None:
    folder_2024 = Path(FOLDER_2024)
    if not folder_2024.is_dir():
        warnings.warn(f"2024 folder not found: {folder_2024.resolve()}; profit simulation skipped.")
        w_opt = None  # disable simulation
    else:
        for sym in symbols:
            csv_file = folder_2024 / f"{sym}.csv"
            if not csv_file.is_file():
                warnings.warn(f"Missing 2024 data for {sym}; profit simulation skipped.")
                w_opt = None
                break
            df = pd.read_csv(csv_file)
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
            df = df.sort_values("datetime").set_index("datetime")
            df_2024 = df.loc[df.index.year == 2024]
            if df_2024.empty:
                warnings.warn(f"No 2024 prices for {sym}; profit simulation skipped.")
                w_opt = None
                break
            fin_price[sym] = float(df_2024["close"].iloc[-1])

# ------------------------------------------------------------
# 6)  Pretty printing
# ------------------------------------------------------------
print("======== Annual μ and σ (percent) ========")
print(summary.round(4).to_string())


if w_opt is not None:
    print("\n======== Optimal Portfolio (σ ≤ {:.0%}) ========".format(RISK_TARGET))
    for s, wt in zip(symbols, w_opt):
        print(f"{s:<15}: {wt:6.2%}")
    print(f"\nμ*  (expected return): {ret_opt:6.2%}")
    print(f"σ*  (portfolio risk) : {sig_opt:6.2%}")

    # ----- Profit calculation -------------------------------------------
    missing_init = [s for s in symbols if s not in init_price]
    missing_fin  = [s for s in symbols if s not in fin_price]
    if missing_init or missing_fin:
        print("\n⚠️  Cannot compute 2023→2024 profit; missing data for:")
        if missing_init:
            print("   • End-2023 close:", ", ".join(missing_init))
        if missing_fin:
            print("   • End-2024 close:", ", ".join(missing_fin))
    else:
        a_shares = {s: CAPITAL_START * w // init_price[s] for s, w in zip(symbols, w_opt)}
        P_end = sum(a_shares[s] * fin_price[s] for s in symbols)
        profit_vnd = P_end - CAPITAL_START
        profit_pct = profit_vnd / CAPITAL_START

        print("\n-------- Investment Simulation --------")
        print("Symbol        Shares       Buy@2023         Sell@2024     Value End-24")
        for s in symbols:
            N = a_shares[s]
            print(f"{s:<12}{N:12.0f}{init_price[s]:15,.0f}{fin_price[s]:15,.0f}{N*fin_price[s]:15,.0f}")
        print("\nStarting capital : {:,.0f} ".format(CAPITAL_START))
        print("Portfolio value  : {:,.0f} ".format(P_end))
        print("Profit           : {:,.0f}  ({:.2%})".format(profit_vnd, profit_pct))
else:
    print("\n(No optimal portfolio – optimisation step skipped or failed.)")

# End of file
