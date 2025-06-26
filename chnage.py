import pandas as pd
from io import StringIO

# ---- sample data ----------------------------------------------------------
df = pd.read_csv("./VN_1W/TEST.csv", parse_dates=["datetime"])
df["close"] = df["close"] + 100000
df.to_csv("./VN_1W/TEST.csv", index=False)