import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from transfer_entropy import TransferEntropy

plt.style.use('fivethirtyeight')
%matplotlib qt
# %%
eqs = 'SPY DIA IWM ECH EWW EWC EWZ'.split()
fi = 'AGG SHY IEI IEF TLT TIP LQD HYG MBB'.split()
cmdtys = 'GLD SLV'.split()

stockdf = pd.read_csv('../data/equities.csv',
            usecols=['Dates']+eqs,
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True)

bonddf = pd.read_csv('../data/fixed_income.csv',
            usecols=['Dates']+fi,
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True)

cmdtydf = pd.read_csv('../data/commodities.csv',
            usecols=['Dates']+cmdtys,
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True)

df = stockdf.join(bonddf).join(cmdtydf)[eqs+fi+cmdtys]
# %%
start = '1/1/2010'
end = '1/1/2019'

# %%
mod = TransferEntropy(data=df)
mod.set_timeperiod(start, end)
mod.compute_effective_transfer_entropy()
te_mat = mod.ete.values.copy()
te_mat_nz = mod.ete.values.copy()
n, m = te_mat_nz.shape
for i in range(n):
    for j in range(m):
        if te_mat_nz[i, j] < 0:
            te_mat_nz[i, j] = 0


prices = df.loc[start:end].resample('M').last()
returns = (prices/prices.shift(1).values-1).dropna()
cov = np.cov(returns, rowvar=False)
rets = returns.mean().values.reshape(-1, 1)
# %%
# rets = np.matrix('.1; .12; .07')
# cov = np.matrix('0.06, 0.0377, 0.0259; 0.0377, 0.095, 0.0285; 0.0259, 0.0285, 0.0287')

n = cov.shape[0]
sims = 1000000

port_rets = np.zeros(sims)
port_vols = np.zeros(sims)
port_tes = np.zeros(sims)
port_tes_nz = np.zeros(sims)

for sim in tqdm(range(sims)):
    w = np.random.uniform(-0.5, 0.5, size=(n, 1))
    w = w/w.sum()

    r = np.squeeze(w.T @ rets)*12
    vol = np.sqrt(np.squeeze(w.T @ cov @ w)*12)
    te = np.squeeze(w.T @ te_mat @ w)
    te_nz = np.squeeze(w.T @ te_mat_nz @ w)
    if vol <= 0.3:
        port_rets[sim] = r
        port_vols[sim] = vol
        port_tes[sim] = te
        port_tes_nz[sim] = te_nz
sr = port_rets/port_vols
# %%
plt.figure(figsize=(10, 6))
plt.scatter(port_vols, port_rets,
    c=sr, cmap='RdYlGn',
    alpha=0.75, s=8)
cbar = plt.colorbar()
cbar.ax.set_title('Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.show()
# %%
plt.figure(figsize=(10, 6))
plt.scatter(port_tes, port_rets,
    c=sr, cmap='RdYlGn',
    alpha=0.75, s=8)
cbar = plt.colorbar()
cbar.ax.set_title('Sharpe Ratio')
# plt.xlim(-0.1, 0.1)
plt.xlabel('Entropy')
plt.ylabel('Return')
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.scatter(port_tes_nz, port_rets,
    c=sr, cmap='RdYlGn',
    alpha=0.75, s=8)
cbar = plt.colorbar()
cbar.ax.set_title('Sharpe Ratio')
# plt.xlim(-0.1, 0.1)
plt.xlabel('Entropy')
plt.ylabel('Return')
plt.show()

# %%
plt.figure(figsize=(10, 6))
plt.scatter(port_vols, port_tes,
    c=sr, cmap='RdYlGn',
    alpha=0.75, s=8)
cbar = plt.colorbar()
cbar.ax.set_title('Sharpe Ratio')
# plt.ylim(-0.002, 0.0015)
plt.xlabel('Volatility')
plt.ylabel('Entropy')
plt.show()
# %%
plt.figure(figsize=(10, 6))
plt.scatter(port_vols, port_tes_nz,
    c=sr, cmap='RdYlGn',
    alpha=0.75, s=8)
cbar = plt.colorbar()
cbar.ax.set_title('Sharpe Ratio')
plt.xlim(None, 0.15)
plt.ylim(-0.02, 0.02)
plt.xlabel('Volatility')
plt.ylabel('Entropy')
plt.show()
