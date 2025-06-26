# #i have folder VN_1W which contain many csv file
# #for each csv file, it have this format:
# #datetime,symbol,open,high,low,close,volume
# # 2020-12-21 03:00:00,HOSE:FPT,31396.3639927,32108.67242534,30903.22738548,31560.74286177,21700818.48199048
# #you should only care about close price
# #now i want to get the average value of the (new close price - the old close price)/old_close_price for each week
# #use that average devide by 1/52 (because we want to estimate the EV of return for each year)
# #that is estimate of mu
# #now for the sigma:
# #\[
# # \bar{\sigma}_n(\Delta t) \;=\;
# # \sqrt{\frac{1}{n}\sum_{i=1}^{n}\bigl[R_i-\bar{R}(\Delta t)\bigr]^2}
# # \]
# # then use it devide by sqrt(delta t) to get the estimate of sigma (risk)
# #remember all the unit is percentage, not vnd or usd, ...
# #finally print the file name with the mu and sigma estimate for each file
# #you can edit this code because this code unit is vnd not percentage:
# import pandas as pd
# import glob
# import os
# import math

# # Path to the folder containing weekly CSV files
# folder = 'VN_1W'

# data = []

# # Iterate over each CSV file in the folder
# for filepath in glob.glob(os.path.join(folder, '*.csv')):
#     try:
#         # Read CSV
#         df = pd.read_csv(filepath)
        
#         # Ensure datetime is sorted (ascending)
#         df = df.sort_values('datetime')
        
#         # Compute weekly difference in close price
#         diff = df['close'].diff().dropna()
        
#         # Mean of weekly differences
#         mu_weekly = diff.mean()
        
#         # Annualised expected return: divide by (1/52)  <=> multiply by 52
#         mu_annual = mu_weekly * 52
        
#         # Weekly standard deviation (population)
#         sigma_weekly = diff.std(ddof=0)
        
#         # Annualised sigma: divide by sqrt(1/52)  <=> multiply by sqrt(52)
#         sigma_annual = sigma_weekly * math.sqrt(52)
        
#         data.append({
#             'file': os.path.basename(filepath),
#             'mu_estimate': mu_annual,
#             'sigma_estimate': sigma_annual
#         })
#     except Exception as e:
#         data.append({
#             'file': os.path.basename(filepath),
#             'mu_estimate': None,
#             'sigma_estimate': None,
#             'error': str(e)
#         })

# # Create a DataFrame with results
# result_df = pd.DataFrame(data)

# # Also print the DataFrame as plain text
# print(result_df)



import pandas as pd
import glob, os, math

# Folder that holds the weekly bars
folder = "VN_1W"

results = []

for filepath in glob.glob(os.path.join(folder, "*.csv")):
    try:
        df = pd.read_csv(filepath)
        df = df.sort_values("datetime")           # make sure rows are in time order

        # ―― WEEK-OVER-WEEK SIMPLE RETURN ( % ) ――――――――――――――――――――――――――――
        # R_i = (Close_i – Close_{i-1}) / Close_{i-1}
        # multiply by 100 at the very end so everything is expressed in % units
        weekly_ret = df["close"].pct_change().dropna()

        # Expected weekly return (μ_Δt) and its stdev (σ_Δt)
        mu_weekly    = weekly_ret.mean()                # E[R]
        sigma_weekly = weekly_ret.std(ddof=0)           # population stdev

        # ―― ANNUALISE (Δt = 1 week = 1/52 yr) ―――――――――――――――――――――――――――――
        mu_annual    = mu_weekly  * 52        * 100     # divide by 1/52 ⇒ ×52, then %  
        sigma_annual = sigma_weekly * math.sqrt(52) * 100  # divide by √(1/52) ⇒ ×√52, then %

        results.append({
            "file": os.path.basename(filepath),
            "mu_%_per_year":    mu_annual,
            "sigma_%_per_year": sigma_annual,
        })

    except Exception as err:
        results.append({
            "file": os.path.basename(filepath),
            "mu_%_per_year":    None,
            "sigma_%_per_year": None,
            "error": str(err),
        })

# Nicely formatted summary
out = pd.DataFrame(results)
print(out.to_string(index=False,
                    col_space=18,
                    float_format="%.4f"))
#for mu and sigma 