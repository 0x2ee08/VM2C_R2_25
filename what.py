# import pandas as pd
# import numpy as np
# import glob
# import os

# # Path to the folder containing weekly CSV files
# folder = 'VN_1W'

# returns_dict = {}
# all_dates = None

# # Read each CSV and compute weekly return (∆close)
# for filepath in glob.glob(os.path.join(folder, '*.csv')):
#     try:
#         df = pd.read_csv(filepath)

#         # Ensure datetime parsed & sorted
#         df['datetime'] = pd.to_datetime(df['datetime'])
#         df = df.sort_values('datetime').set_index('datetime')

#         # Symbol (prefer column; else use filename)
#         symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else os.path.splitext(os.path.basename(filepath))[0]

#         # Weekly difference of close price
#         returns = df['close'].diff().dropna()

#         returns_dict[symbol] = returns

#         # collect dates
#         all_dates = returns.index if all_dates is None else all_dates.union(returns.index)

#     except Exception as e:
#         print(f"Error processing {filepath}: {e}")

# # Create aligned DataFrame of returns
# returns_df = pd.DataFrame(index=sorted(all_dates))
# for sym, series in returns_dict.items():
#     returns_df[sym] = series

# # Drop rows where all stocks NaN
# returns_df = returns_df.dropna(how='all')

# # Calculate weekly μ̄ and σ̄ (population)
# mu_series = returns_df.mean(skipna=True)           # mean across rows
# sigma_series = returns_df.std(ddof=0, skipna=True) # population std

# # Population covariance matrix
# cov_matrix = returns_df.cov(ddof=0)

# # Population correlation matrix ρ = cov / (σ_i σ_j)
# outer_sigma = np.outer(sigma_series, sigma_series)
# corr_matrix = cov_matrix / outer_sigma
# corr_df = pd.DataFrame(corr_matrix, index=cov_matrix.index, columns=cov_matrix.columns)

# # Prepare summary μ & σ table
# summary_df = pd.DataFrame({
#     'mu_weekly': mu_series,
#     'sigma_weekly': sigma_series
# })

# # Display results to the us
# print(corr_df)

import pandas as pd
import numpy as np
import glob, os

folder = "VN_1W"

returns_dict = {}
all_dates = None

for filepath in glob.glob(os.path.join(folder, "*.csv")):
    try:
        df = pd.read_csv(filepath)

        # --- tidy & index ----------------------------------------------------
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").set_index("datetime")

        symbol = df["symbol"].iloc[0] if "symbol" in df.columns \
                 else os.path.splitext(os.path.basename(filepath))[0]

        # --- WEEKLY % RETURN --------------------------------------------------
        # R_t  = (Close_t − Close_{t-1}) / Close_{t-1}  × 100  (simple %)
        returns = df["close"].pct_change().mul(100).dropna()

        returns_dict[symbol] = returns
        all_dates = returns.index if all_dates is None else all_dates.union(returns.index)

    except Exception as e:
        print(f"Error processing {filepath}: {e}")

# ------------------- ALIGN ALL SERIES ON THE SAME DATE INDEX -----------------
returns_df = pd.DataFrame(index=sorted(all_dates))
for sym, series in returns_dict.items():
    returns_df[sym] = series.reindex(returns_df.index)

returns_df = returns_df.dropna(how="all")      # toss rows where every symbol is NaN
print(returns_df)
# ------------------- SUMMARY STATISTICS (all now in % units) ------------------
mu_series    = returns_df.mean(skipna=True)          # expected weekly return  (μ̄)    %
sigma_series = returns_df.std(ddof=0, skipna=True)   # weekly stdev           (σ̄)    %

returns_df = returns_df.div(100);
cov_matrix   = returns_df.cov(ddof=0) * 52                # %²


summary_df = pd.DataFrame({
    "mu_weekly_%":    mu_series,
    "sigma_weekly_%": sigma_series
})

# ------------------- OUTPUT ---------------------------------------------------
print("Cov matrix ():")
print(cov_matrix.round(4))
#for corr