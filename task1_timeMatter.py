"""
VN Weekly Analysis — Exponential-Decay Version (customisable memory)
===================================================================

Same workflow as before but with a **softer, user-friendly time-decay knob**:
choose how much you still want to “remember” after a given horizon and the
script computes the matching half-life automatically.

Key idea
--------
If you say *“after 26 weeks I still want 90 % of today’s weight”* then the decay
per week should be λ = 0.9**(1/26).  Exponential decay is memory-less, so after
_n_ weeks an observation’s relative weight is λⁿ.

Formula
```
λ = retention_ratio ** (1 / retention_weeks)
HALFLIFE = ln(0.5) / ln(λ)
```

You only have to set two intuitive numbers:
    •  `RETENTION_RATIO`  (0–1)  – remaining weight after *RETENTION_WEEKS*
    •  `RETENTION_WEEKS`          – the horizon in weeks

Everything else is computed.
"""

from __future__ import annotations

import glob
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Parameters — tweak here if you like
# ------------------------------------------------------------
FOLDER_1W:     Path | str = "VN_1W"          # weekly bars, 2021-2023
FOLDER_2024:   Path | str = "VN_2024"        # 2024 data (any frequency)
ANNUAL_FACTOR: int        = 52               # 1 week = 1/52 year
RISK_TARGET:   float      = 0.20             # 20 % annual volatility cap
ALLOW_SHORT:   bool       = False            # set True to allow short sales
CAPITAL_START: float      = 26_169_995_100.0 # VND 25 billion @ 2023-12-31

# --- Time-decay settings ------------------------------------
RETENTION_RATIO: float = 0.90   # keep 90 % of the weight …
RETENTION_WEEKS: int   = 26     # … after 26 weeks
λ = RETENTION_RATIO ** (1 / RETENTION_WEEKS)
HALFLIFE_WEEKS: float = math.log(0.5) / math.log(λ)  # derived ≈ 171 weeks
# If you prefer to specify the half-life directly, just set HALFLIFE_WEEKS and
# comment the two lines above.  All maths below relies only on HALFLIFE_WEEKS.

# ------------------------------------------------------------
# 1)  Load weekly files and build a returns DataFrame
# ------------------------------------------------------------
folder_path = Path(FOLDER_1W)
if not folder_path.is_dir():
    raise SystemExit(f"Folder not found: {folder_path.resolve()}")

returns_dict: dict[str, pd.Series] = {}
init_price:   dict[str, float]   = {}
all_dates:    pd.DatetimeIndex | None = None

for csv_file in glob.glob(str(folder_path / "*.csv")):
    try:
        df = pd.read_csv(csv_file)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime").set_index("datetime")

        # drop rows with NaT indices (in-line to keep memory low)
        if df.index.hasnans:
            warnings.warn(f"Missing datetime in {csv_file}; dropped affected rows")
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
            weekly_pct.index if all_dates is None else all_dates.union(weekly_pct.index)
        )

    except Exception as exc:
        warnings.warn(f"⚠️  Skipping {csv_file}: {exc}")

if not returns_dict:
    raise SystemExit("No valid CSV files were processed.")

# Align series on a common calendar so matrix maths works nicely
returns_df = pd.DataFrame(index=sorted(all_dates))
for sym, ser in returns_dict.items():
    returns_df[sym] = ser.reindex(returns_df.index)

# Drop weeks where every symbol is NaN (e.g. market holidays)
returns_df.dropna(how="all", inplace=True)

# ------------------------------------------------------------
# 2)  Per-symbol μ and σ (EWMA, still in percent here)
# ------------------------------------------------------------
ewm_stats = returns_df.ewm(halflife=HALFLIFE_WEEKS, adjust=False)
mu_weekly_pct    = ewm_stats.mean().iloc[-1]          # Series (last timestamp)
sigma_weekly_pct = ewm_stats.var().iloc[-1].pow(0.5)  # σ = √Var

mu_annual_pct    = mu_weekly_pct    * ANNUAL_FACTOR
sigma_annual_pct = sigma_weekly_pct * math.sqrt(ANNUAL_FACTOR)

summary = pd.DataFrame({
    "mu_annual_%":    mu_annual_pct,
    "sigma_annual_%": sigma_annual_pct,
}).sort_index()

# ------------------------------------------------------------
# 3)  Correlation & Covariance matrices (EWMA)
# ------------------------------------------------------------
returns_dec_weekly = returns_df.div(100)  # % → decimal
Σ_weekly = (
    returns_dec_weekly
      .ewm(halflife=HALFLIFE_WEEKS, adjust=False)
      .cov(pairwise=True)
      .xs(returns_dec_weekly.index[-1], level=0)
)
Σ_annual = Σ_weekly * ANNUAL_FACTOR

# Correlation uses the same decay profile
std_vec = np.sqrt(np.diag(Σ_weekly))
with np.errstate(invalid="ignore", divide="ignore"):
    ρ = Σ_weekly.div(std_vec, axis=0).div(std_vec, axis=1)

# ------------------------------------------------------------
# 4)  Markowitz optimisation (long-only unless ALLOW_SHORT=True)
# ------------------------------------------------------------
try:
    import gurobipy as gp
    from gurobipy import GRB

    symbols = summary.index.to_list()
    μ_vec = summary["mu_annual_%"].values / 100  # decimal form

    m = gp.Model("Markowitz")
    w = m.addVars(len(symbols), lb=0.0 if not ALLOW_SHORT else -GRB.INFINITY,
                  name="w")

    # Maximise expected return (Gurobi minimises by default)
    m.setObjective(-gp.quicksum(μ_vec[i] * w[i] for i in range(len(symbols))),
                   GRB.MINIMIZE)

    # Budget constraint  ∑w = 1
    m.addConstr(gp.quicksum(w[i] for i in range(len(symbols))) == 1, "budget")

    # Risk constraint  wᵀ Σ w ≤ σ_max²
    quad = gp.QuadExpr()
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            if Σ_annual.iat[i, j]:  # skip zeros to speed‑up
                quad.add(Σ_annual.iat[i, j] * w[i] * w[j])
    m.addQConstr(quad <= RISK_TARGET ** 2, "risk")

    m.Params.OutputFlag = 0  # mute solver log
    m.optimize()

    if m.Status == GRB.OPTIMAL:
        w_opt   = np.array([w[i].X for i in range(len(symbols))])
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
# 5)  Profit simulation 2023‑12‑31 → 2024‑12‑31
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
print("======== Annual μ and σ (percent, EWMA) ========")
print(summary.round(4).to_string())

print("\n======== Weekly Correlation Matrix ρ (EWMA) ========")
print(ρ.round(4).to_string())

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
            print("   • End‑2023 close:", ", ".join(missing_init))
        if missing_fin:
            print("   • End‑2024 close:", ", ".join(missing_fin))
    else:
        a_shares = {s: CAPITAL_START * w // init_price[s] for s, w in zip(symbols, w_opt)}
        P_end = sum(a_shares[s] * fin_price[s] for s in symbols)
        profit_vnd = P_end - CAPITAL_START
        profit_pct = profit_vnd / CAPITAL_START

        print("\n-------- Investment Simulation --------")
        print("Symbol        Shares       Buy@2023         Sell@2024     Value End‑24")
        for s in symbols:
            N = a_shares[s]
            print(f"{s:<12}{N:12.0f}{init_price[s]:15,.0f}{fin_price[s]:15,.0f}{N*fin_price[s]:15,.0f}")
        print("\nStarting capital : {:,.0f} ".format(CAPITAL_START))
        print("Portfolio value  : {:,.0f} ".format(P_end))
        print("Profit           : {:,.0f}  ({:.2%})".format(profit_vnd, profit_pct))
else:
    print("\n(No optimal portfolio – optimisation step skipped or failed.)")

# End of file
