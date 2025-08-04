# ------------------------------
# Clean Sector Rotation vs. SPY (Single Cell)
# ------------------------------

# Install & Pin

# Clear any past outputs & figures
import matplotlib.pyplot as plt
plt.close('all')

# Imports
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm

# Params
ETFS   = ["XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLK","XLU"]
START  = "2015-01-01"
MOM    = 6       # months lookback
TOP_N  = 3       # of ETFs to hold
TC     = 0.001   # .1% round-trip cost

# Download & Prep Data
prices   = yf.download(ETFS, start=START, auto_adjust=True)["Close"]
rets     = prices.pct_change().fillna(0)
price_m  = prices.resample("ME").last()                   # month-end
mom      = price_m / price_m.shift(MOM) - 1
rank_pct = mom.rank(pct=True, axis=1)
signal_m = (rank_pct >= (1 - TOP_N/len(ETFS))).astype(float)
pos      = signal_m.reindex(prices.index, method="ffill")
pos      = pos.div(pos.sum(axis=1), axis=0).fillna(0)

# Strategy & Benchmark
strategy_returns = (pos.shift() * rets).sum(axis=1)
strategy_returns -= pos.diff().abs().sum(axis=1) * TC
cumulative_strategy = (1 + strategy_returns).cumprod()

spy       = yf.download("SPY", start=START, auto_adjust=True)["Close"]
spy_ret   = spy.pct_change().fillna(0)
cum_spy   = (1 + spy_ret).cumprod().reindex(cumulative_strategy.index, method="ffill")

# Plot
# Plot
plt.figure(figsize=(10,5))
plt.plot(cumulative_strategy, label="6-M Sector Rotation")  # Explicitly use plt.plot()
plt.plot(cum_spy, label="SPY Buy-Hold")
plt.title("6-Month Momentum Sector Rotation vs. SPY")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.show()

# Metrics
def perf_stats(r, cr):
    a = np.asarray(r.dropna())
    if a.size == 0: return (np.nan,)*4
    ann = a.mean()*252
    vol = a.std(ddof=0)*np.sqrt(252)
    sr  = ann/vol if vol!=0 else np.nan
    dd  = np.max(np.maximum.accumulate(cr)-cr)
    return ann, vol, sr, dd

s_r, s_v, s_sr, s_dd = perf_stats(strategy_returns, cumulative_strategy)
p_r, p_v, p_sr, p_dd = perf_stats(spy_ret.loc[cumulative_strategy.index], cum_spy)

print(f"{'Metric':20}{'Rotation':>12}{'SPY':>12}")
print("-"*44)
print(f"{'Ann. Return':20}{s_r:12.2%}{p_r:12.2%}")
print(f"{'Ann. Volatility':20}{s_v:12.2%}{p_v:12.2%}")
print(f"{'Sharpe Ratio':20}{s_sr:12.2f}{p_sr:12.2f}")
print(f"{'Max Drawdown':20}{s_dd:12.2%}{p_dd:12.2%}")

# ------------------------------

# ------------------------------

# Installs & Imports

, yfinance as yf, matplotlib.pyplot as plt
from tqdm import tqdm

plt.close('all')

# Params (same as before)
ETFS   = ["XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLK","XLU"]
START  = "2015-01-01"
MOM    = 6
TOP_N  = 3
TC     = 0.001

# Download data
prices = yf.download(ETFS + ["SPY"], start=START, auto_adjust=True)["Close"]
rets   = prices.pct_change().fillna(0)

# Build sector rotation signal
price_m    = prices[ETFS].resample("ME").last()
mom        = price_m / price_m.shift(MOM) - 1
rank_pct   = mom.rank(pct=True, axis=1)
signal_m   = (rank_pct >= (1 - TOP_N/len(ETFS))).astype(float)
pos_sec    = signal_m.reindex(prices.index, method="ffill")
pos_sec    = pos_sec.div(pos_sec.sum(axis=1), axis=0).fillna(0)

# Compute bull/bear regime from SPY 200-day MA
spy_price  = prices["SPY"]
ma200      = spy_price.rolling(200).mean()
bull_reg   = (spy_price > ma200).astype(float)   # =bull, 0=bear

# Build regime-adaptive daily weights
# In bull: sector rotation;  In bear: 100% SPY
pos_reg     = pos_sec.multiply(bull_reg, axis=0)
pos_reg["SPY"] = 1 - bull_reg   # when bear, SPY=1; when bull, SPY=0

# Backtest each strategy
def backtest(pos_df, returns_df):
    strat = (pos_df.shift() * returns_df).sum(axis=1)
    # subtract costs only on sector legs
    tc = pos_df[ETFS].diff().abs().sum(axis=1) * TC
    return (strat - tc).fillna(0).rename("daily_returns")

# sector‐only rotation
rotation_daily_returns  = backtest(pos_sec.join( pd.DataFrame(0, index=pos_sec.index, columns=["SPY"])), rets)
cumulative_rotation  = (1 + rotation_daily_returns).cumprod()

# regime‐adaptive rotation
regime_daily_returns  = backtest(pos_reg, rets)
cumulative_regime  = (1 + regime_daily_returns).cumprod()

# SPY buy‐hold
spy_ret  = rets["SPY"]
cum_spy  = (1 + spy_ret).cumprod()

# Plot all three
plt.figure(figsize=(10,5))
cumulative_rotation.plot(label="Rotation (always)")
cumulative_regime.plot(label="Rotation w/ Regime Filter")
cum_spy.plot(label="SPY Buy-Hold")
# shade bear periods
bear = bull_reg==0
plt.fill_between(cumulative_regime.index, 0, 5, where=bear, color='gray', alpha=0.2, transform=plt.gca().get_xaxis_transform())
plt.title("Regime-Adaptive Sector Rotation vs. SPY")
plt.ylabel("Cumulative Return"); plt.xlabel("Date")
plt.legend(); plt.grid(True)
plt.show()

# Metrics summary
def perf(r):
    ann = r.mean()*252
    vol = r.std(ddof=0)*np.sqrt(252)
    sr  = ann/vol if vol else np.nan
    dd  = np.max(np.maximum.accumulate((1+r).cumprod()) - (1+r).cumprod())
    return ann, vol, sr, dd

rows = [
    ("Always Rotation", *perf(rotation_daily_returns)),
    ("Regime-Adaptive", *perf(regime_daily_returns)),
    ("SPY Buy-Hold",    *perf(spy_ret))
]
print(f"{'Strategy':25}{'AnnRet':>8}{'AnnVol':>8}{'Sharpe':>8}{'MaxDD':>8}")
for name, a,v,s,d in rows:
    print(f"{name:25}{a:8.2%}{v:8.2%}{s:8.2f}{d:8.2%}")

# ------------------------------

# ------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Compute avg weights & total contributions
# pos: DataFrame of daily weights for each ETF (from your rotation code)
# rets: DataFrame of daily returns (same ETFS + SPY)
daily_contrib   = pos.shift() * rets[ETFS]         # each ETF’s daily P&L contribution
total_contrib   = daily_contrib.sum()              # sum over time = total return contribution
avg_weights     = pos.mean()                       # average weight over the backtest

attr = pd.DataFrame({
    "avg_weight":    avg_weights,
    "total_return":  total_contrib
}).sort_values("total_return", ascending=False)

print("=== Sector Attribution ===")
print(attr.to_string(float_format="{:,.2%}".format))

# Bar chart of contributions vs. weights
fig, ax1 = plt.subplots(figsize=(8,4))
attr["total_return"].plot.bar(ax=ax1, rot=45, position=0, width=0.4, label="Total Return")
ax2 = ax1.twinx()
(attr["avg_weight"]).plot.bar(ax=ax2, rot=45, position=1, width=0.4, color="C1", label="Avg Weight")
ax1.set_ylabel("Total Return")
ax2.set_ylabel("Avg Weight")
ax1.set_title("Sector Rotation Attribution")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
plt.tight_layout()
plt.show()

# Rolling Beta vs SPY
window = 60  # trading days (~3 months)
beta = strategy_returns.rolling(window).cov(rets["SPY"]) / rets["SPY"].rolling(window).var()
beta.dropna().plot(figsize=(8,3), title=f"{window}-Day Rolling Beta vs SPY")
plt.ylabel("Beta"); plt.grid(True)
plt.show()

, yfinance as yf
from tqdm import tqdm

# Re-define params
ETFS    = ["XLY","XLP","XLE","XLF","XLV","XLI","XLB","XLK","XLU"]
START   = "2015-01-01"
TC      = 0.001   # transaction cost

# Sweep settings
lookbacks = [3, 6, 9, 12]    # months
top_ns    = [2, 3, 4, 5]     # number of ETFs to hold

# Fetch once
prices = yf.download(ETFS, start=START, auto_adjust=True)["Close"]
rets   = prices.pct_change().fillna(0)
price_m = prices.resample("ME").last()

results = []
for mom in tqdm(lookbacks, desc="Lookbacks"):
    # build momentum table for this lookback
    mom_ret   = price_m / price_m.shift(mom) - 1
    rank_pct  = mom_ret.rank(pct=True, axis=1)

    for top_n in top_ns:
        # build signals & positions
        signal_m = (rank_pct >= (1 - top_n/len(ETFS))).astype(float)
        pos      = signal_m.reindex(prices.index, method="ffill")
        pos      = pos.div(pos.sum(axis=1), axis=0).fillna(0)

        # strategy returns net of cost
        strategy_returns = (pos.shift() * rets).sum(axis=1)
        strategy_returns -= pos.diff().abs().sum(axis=1) * TC
        cum       = (1 + strategy_returns).cumprod()

        # performance stats
        ann_ret   = strategy_returns.mean() * 252
        ann_vol   = strategy_returns.std(ddof=0) * np.sqrt(252)
        sharpe    = ann_ret / ann_vol if ann_vol else np.nan
        max_dd    = (cum.cummax() - cum).max()

        results.append({
            "lookback": mom,
            "top_n":    top_n,
            "Ann Return":   ann_ret,
            "Ann Vol":      ann_vol,
            "Sharpe":       sharpe,
            "Max Drawdown": max_dd
        })

# Compile & display
df_res = pd.DataFrame(results)

# Corrected pivot call
df_pivot = df_res.pivot(index="lookback", columns="top_n", values="Sharpe")

print("Sharpe Ratio heatmap (rows=lookback, cols=top_n):")
display(df_pivot)

print("\nTop 10 configs by Sharpe:")
display(df_res.sort_values("Sharpe", ascending=False).head(10))

# ------------------------------

# ------------------------------

import numpy as np
import matplotlib.pyplot as plt

# ) Reconstruct pivot (if not already)
df_pivot = df_res.pivot(index="lookback", columns="top_n", values="Sharpe")

# ) Plot heatmap
fig, ax = plt.subplots(figsize=(6,4))
data = df_pivot.values
cax  = ax.imshow(data, origin="lower", aspect="auto", cmap="viridis")
fig.colorbar(cax, label="Sharpe Ratio")

# ) Tick labels
ax.set_xticks(np.arange(len(df_pivot.columns)))
ax.set_xticklabels(df_pivot.columns)
ax.set_yticks(np.arange(len(df_pivot.index)))
ax.set_yticklabels(df_pivot.index)
ax.set_xlabel("Top N ETFs")
ax.set_ylabel("Momentum Lookback (months)")
ax.set_title("Sharpe Ratio Heatmap")

# ) Annotate each cell with its Sharpe value
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        val = data[i,j]
        ax.text(j, i, f"{val:.2f}",
                ha="center", va="center",
                color="white" if val < data.max()/2 else "black")

# ) Highlight the best config
best = df_res.loc[df_res["Sharpe"].idxmax()]
bi   = df_pivot.index.get_loc(best["lookback"])
bj   = df_pivot.columns.get_loc(best["top_n"])
from matplotlib.patches import Rectangle
rect = Rectangle((bj-0.5, bi-0.5), 1, 1, fill=False, edgecolor="red", linewidth=2)
ax.add_patch(rect)

plt.tight_layout()
plt.show()

# # # #
