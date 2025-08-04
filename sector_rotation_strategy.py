# Clean Sector Rotation vs. SPY

!pip install --quiet pandas==2.2.2 yfinance tqdm matplotlib

from IPython.display import clear_output
import matplotlib.pyplot as plt
clear_output(wait=True)
plt.close('all')

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

# Parameters
ETFS   = ["XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLK", "XLU"]
START  = "2015-01-01"
MOM    = 6
TOP_N  = 3
TC     = 0.001

# Data download and processing
prices   = yf.download(ETFS, start=START, auto_adjust=True)["Close"]
rets     = prices.pct_change().fillna(0)
price_m  = prices.resample("M").last()
mom      = price_m / price_m.shift(MOM) - 1
rank_pct = mom.rank(pct=True, axis=1)
signal_m = (rank_pct >= (1 - TOP_N / len(ETFS))).astype(float)
pos      = signal_m.reindex(prices.index, method="ffill")
pos      = pos.div(pos.sum(axis=1), axis=0).fillna(0)

# Strategy vs. SPY
strat_ret = (pos.shift() * rets).sum(axis=1)
strat_ret -= pos.diff().abs().sum(axis=1) * TC
cum_strat = (1 + strat_ret).cumprod()

spy       = yf.download("SPY", start=START, auto_adjust=True)["Close"]
spy_ret   = spy.pct_change().fillna(0)
cum_spy   = (1 + spy_ret).cumprod().reindex(cum_strat.index, method="ffill")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(cum_strat, label="Sector Rotation")
plt.plot(cum_spy, label="SPY")
plt.title("6-Month Momentum Strategy vs SPY")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()

# Performance metrics
def perf_stats(r, cr):
    daily = np.asarray(r.dropna())
    if daily.size == 0: return (np.nan,)*4
    ann = daily.mean() * 252
    vol = daily.std(ddof=0) * np.sqrt(252)
    sr  = ann / vol if vol != 0 else np.nan
    dd  = np.max(np.maximum.accumulate(cr) - cr)
    return ann, vol, sr, dd

s_r, s_v, s_sr, s_dd = perf_stats(strat_ret, cum_strat)
p_r, p_v, p_sr, p_dd = perf_stats(spy_ret.loc[cum_strat.index], cum_spy)

print(f"{'Metric':20}{'Strategy':>12}{'SPY':>12}")
print("-" * 44)
print(f"{'Annual Return':20}{s_r:12.2%}{p_r:12.2%}")
print(f"{'Volatility':20}{s_v:12.2%}{p_v:12.2%}")
print(f"{'Sharpe Ratio':20}{s_sr:12.2f}{p_sr:12.2f}")
print(f"{'Max Drawdown':20}{s_dd:12.2%}{p_dd:12.2%}")
