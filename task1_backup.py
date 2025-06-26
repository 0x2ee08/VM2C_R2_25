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
FOLDER: Path | str = "VN_1W"   # where the CSVs live
ANNUAL_FACTOR: int = 52        # 1 week = 1/52 year
RISK_TARGET: float = 0.20      # 20 % annual volatility cap
ALLOW_SHORT: bool = False      # set True if you want to allow short sales

# ------------------------------------------------------------
# 1)  Load all files and build a single returns DataFrame
# ------------------------------------------------------------
folder_path = Path(FOLDER)
if not folder_path.is_dir():
    raise SystemExit(f"Folder not found: {folder_path.resolve()}")

returns_dict: dict[str, pd.Series] = {}
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

# ------------------------------------------------------------
# 2)  Per‑symbol μ and σ (all still in percent here)
# ------------------------------------------------------------
mu_weekly_pct = returns_df.mean(skipna=True)
σ_weekly_pct = returns_df.std(ddof=0, skipna=True)  # population stdev

mu_annual_pct = mu_weekly_pct * ANNUAL_FACTOR
σ_annual_pct = σ_weekly_pct * math.sqrt(ANNUAL_FACTOR)

summary = pd.DataFrame({
    "mu_annual_%": mu_annual_pct,
    "sigma_annual_%": σ_annual_pct,
}).sort_index()

# ------------------------------------------------------------
# 3)  Correlation & Covariance matrices
# ------------------------------------------------------------
ρ = returns_df.corr()  # already unit‑less because both inputs are %

# Convert weekly % returns → decimal, then annualise covariance
returns_dec_weekly = returns_df.div(100)            # % → decimal
Σ_annual = returns_dec_weekly.cov(ddof=0) * ANNUAL_FACTOR  # var scales linearly with time

# ------------------------------------------------------------
# 4)  Markowitz optimisation (long‑only unless ALLOW_SHORT=True)
# ------------------------------------------------------------
try:
    import gurobipy as gp
    from gurobipy import GRB

    symbols = summary.index.to_list()
    μ_vec = summary["mu_annual_%"].values / 100              # decimal form

    # Gurobi setup
    m = gp.Model("Markowitz")
    w = m.addVars(len(symbols), lb=0.0 if not ALLOW_SHORT else -GRB.INFINITY, name="w")

    # Maximise expected return (note: Gurobi minimises by default)
    m.setObjective(-gp.quicksum(μ_vec[i] * w[i] for i in range(len(symbols))), GRB.MINIMIZE)

    # Budget constraint  ∑w = 1
    m.addConstr(gp.quicksum(w[i] for i in range(len(symbols))) == 1, "budget")

    # Risk constraint  wᵀ Σ w ≤ σ_max²
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

# ------------------------------------------------------------
# 5)  Pretty printing
# ------------------------------------------------------------
print("======== Annual μ and σ (percent) ========")
print(summary.round(4).to_string())

print("\n======== Weekly Correlation Matrix ρ ========")
print(ρ.round(4).to_string())

if w_opt is not None:
    print("\n======== Optimal Portfolio (σ ≤ {:.0%}) ========".format(RISK_TARGET))
    for s, wt in zip(symbols, w_opt):
        print(f"{s:<15}: {wt:6.2%}")
    print(f"\nμ*  (expected return): {ret_opt:6.2%}")
    print(f"σ*  (portfolio risk) : {sig_opt:6.2%}")
else:
    print("\n(No optimal portfolio – optimisation step skipped or failed.)")

# End of file
