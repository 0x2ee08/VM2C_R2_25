# ===================================================================
# Markowitz: max μᵀ w   s.t.  wᵀ Σ w ≤ σ_max² ,  ∑w = 1 ,  w ≥ 0
# ===================================================================

import numpy as np
import gurobipy as gp
from gurobipy import GRB

# ---------------------------------------------------------------
# 1)  INPUT DATA  (all in DECIMAL form, i.e. 0.08 = 8 %)
# ---------------------------------------------------------------
symbols = [
    "HOSE:VIC", "HOSE:REE", "HOSE:GAS", "HOSE:HPG", "HOSE:MWG",
    "HOSE:VCB", "HOSE:VNM", "HOSE:PNJ", "HOSE:FPT", "HOSE:VJC"
]

# Annual expected returns μ  (6.28 % → 0.062845, etc.)
mu = np.array([
     0.062845,  0.250702,  0.023454,  0.110701,  0.161816,
     0.085719, -0.134934,  0.099542,  0.327986, -0.052170
])

# Annual volatilities σ  (33.9786 % → 0.339786, etc.)
sigma = np.array([
    0.339786, 0.302875, 0.293778, 0.357178, 0.362692,
    0.234905, 0.191564, 0.284328, 0.267271, 0.218353
])

# Correlation matrix ρ  (same order as `symbols`)
corr = np.array([
    [1.0000, 0.1252, 0.1617, 0.2544, 0.0845, 0.1007, 0.0770, 0.0904, 0.1017, 0.1186],
    [0.1252, 1.0000, 0.4128, 0.3154, 0.4240, 0.2151, 0.2313, 0.3889, 0.4934, 0.1344],
    [0.1617, 0.4128, 1.0000, 0.2965, 0.3436, 0.3516, 0.1720, 0.3336, 0.3857, 0.1125],
    [0.2544, 0.3154, 0.2965, 1.0000, 0.4224, 0.4392, 0.2536, 0.3199, 0.3628, 0.2160],
    [0.0845, 0.4240, 0.3436, 0.4224, 1.0000, 0.3889, 0.2896, 0.4520, 0.4841, 0.1945],
    [0.1007, 0.2151, 0.3516, 0.4392, 0.3889, 1.0000, 0.3427, 0.3014, 0.3436, 0.1733],
    [0.0770, 0.2313, 0.1720, 0.2536, 0.2896, 0.3427, 1.0000, 0.3271, 0.4298, 0.1590],
    [0.0904, 0.3889, 0.3336, 0.3199, 0.4520, 0.3014, 0.3271, 1.0000, 0.5144, 0.2530],
    [0.1017, 0.4934, 0.3857, 0.3628, 0.4841, 0.3436, 0.4298, 0.5144, 1.0000, 0.2249],
    [0.1186, 0.1344, 0.1125, 0.2160, 0.1945, 0.1733, 0.1590, 0.2530, 0.2249, 1.0000]
])

# Covariance matrix Σ = σ σᵀ ⊙ ρ
Sigma = np.outer(sigma, sigma) * corr

n = len(symbols)
risk_target = 0.20           # 20 % annual volatility
allow_short = False          # no short sales

# ---------------------------------------------------------------
# 2)  BUILD & SOLVE WITH GUROBI
# ---------------------------------------------------------------
m = gp.Model("Markowitz_pct")
w = m.addVars(n,
              lb=0.0 if not allow_short else -GRB.INFINITY,
              name="w")

# Maximise μᵀ w   (Gurobi minimises by default → minimise the negative)
m.setObjective(-gp.quicksum(mu[i] * w[i] for i in range(n)), GRB.MINIMIZE)

# ∑ w = 1  (full budget)
m.addConstr(gp.quicksum(w[i] for i in range(n)) == 1, "budget")

# Risk: wᵀ Σ w ≤ σ_max²
quad = gp.QuadExpr()
for i in range(n):
    for j in range(n):
        if corr[i, j] != 0:                # very minor speed-up
            quad.add(Sigma[i, j] * w[i] * w[j])
m.addQConstr(quad <= risk_target**2, "risk")

m.Params.OutputFlag = 0                   # set to 1 if you want the solver log
m.optimize()

# ---------------------------------------------------------------
# 3)  RESULTS
# ---------------------------------------------------------------
if m.Status == GRB.OPTIMAL:
    w_opt   = np.array([w[i].X for i in range(n)])
    ret_opt = float(mu @ w_opt)
    sig_opt = float(np.sqrt(w_opt @ Sigma @ w_opt))
    print("------------ Optimal Portfolio (Gurobi) -------------")
    for s, wt in zip(symbols, w_opt):
        print(f"{s:<8}: {wt:6.2%}")
    print(f"\nExpected return μ* : {ret_opt:6.2%}")
    print(f"Portfolio σ*       : {sig_opt:6.2%}")
    print(f"Risk target σ_max  : {risk_target:6.2%}")
    #calculate profit
    #we have P_start = 25_000_000_000 vnd
    #S_k_start will be the close price in the end of 2023
    #will we calculate a_k = P_start*w_k/S_k_start
    #a_k is the number of stock of S_k we will buy
    #P_end = sigma(a_k*S_k_end)
    #S_k_end is the close price in the end of 2024 which can be found in VN_2024 folder with alot of csv record for that year
    #note VN_1W folder only contain record from 2021 to end of 2023
else:
    print("No optimal solution found. Status:", m.Status)
