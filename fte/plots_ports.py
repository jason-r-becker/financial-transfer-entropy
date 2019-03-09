import time
from collections import defaultdict
from datetime import timedelta

import cvxpy as cp
import empiricalutilities as eu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from transfer_entropy import TransferEntropy

plt.style.use('fivethirtyeight')
# %%
eqs = 'SPY XLK XLV XLF IYZ XLY XLP XLI XLE XLU IYR XLB'\
    ' DIA IWM ECH EWW EWC EWZ'.split()
fi = 'AGG SHY IEI IEF TLT TIP LQD HYG MBB'.split()
cmdtys = 'GLD SLV DBA DBC USO UNG'.split()
assets = eqs + fi + cmdtys

def cum_rets(rets):
    cum_rets = []
    cum_rets.append(1)
    for i, ret in enumerate(rets):
        cum_rets.append(cum_rets[i]*(1+ret))
    return cum_rets

# %%
q = 4
res = {}
ete_mats = {}
mod = TransferEntropy(assets=assets)
period = '6M'
months = mod.prices.index.to_period(period).unique().to_timestamp()
iters = len(months)-1

with tqdm(total=iters) as pbar:
    # for start, end in zip(months[:-1], months[1:]):
    for start, end in zip(['2010-01-01', '2010-07-01'], ['2010-07-01', '2011-01-01']):
        # end -= timedelta(1)
        mod.set_timeperiod(start, end)
        mod.compute_effective_transfer_entropy(sims=50, bins=6,
                                               std_threshold=0)
        ete = mod.ete.copy()
        ete_out = ete.sum(axis=0)

        returns = mod.prices.iloc[-1]/mod.prices.iloc[0]-1
        vols = mod.data.std()
        df = pd.DataFrame({'returns': returns, 'vol': vols, 'ete': ete_out})
        df['q'] = pd.qcut(ete_out, q=q, labels=False)

        res[start] = df.groupby('q').agg('mean').reset_index().copy()
        ete_mats[start] = ete
        pbar.update(1)

ete_df = pd.concat(ete_mats)
ete_df.to_csv(f'../ete_{period}.csv')
(ete_mats['2010-01-01']==0).sum().sum()/(33**2)
# %%
q_rets = {}
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
cmap = sns.color_palette('Blues_r', n_colors=4)

resdf = pd.concat(res)
resdf.index = resdf.index.droplevel(1)

for c, qtile in zip(cmap, range(q)):
    q_rets[qtile] = resdf[resdf['q']==qtile]['returns'].values
    ax.plot(months, cum_rets(q_rets[qtile]), c=c,
        lw=2, alpha=1, label=f'Quartile {qtile+1}')

fig.autofmt_xdate()
plt.ylabel('Cumulative Return')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
plt.savefig(f'../plots/quartile_returns.png', dpi=300)
eu.latex_figure(f'../data/plots/quartile_returns.png')
# %%
table = defaultdict(dict)
for qtile in range(q):
    table[qtile]['r'] = resdf[resdf['q']==qtile]['returns'].mean()*12
    table[qtile]['v'] = resdf[resdf['q']==qtile]['returns'].std()*np.sqrt(12)
    table[qtile]['ete'] = resdf[resdf['q']==qtile]['ete'].mean()

table = pd.DataFrame.from_dict(table, orient='index')
table['sr'] = table['r']/table['v']
table = table.reset_index()
table = table['index r v sr ete'.split()]
table.columns = 'Quartile Return Volatility Sharpe ETE'.split()
table['Quartile'] += 1
table[['Return', 'Volatility']] *= 100
eu.latex_print(table, prec=2, hide_index=True)
# %%
def get_CAPM_weights(er, cov, gamma):
    n = cov.shape[0]
    w = cp.Variable((n, 1))
    gamma = cp.Parameter(nonneg=True, value=gamma)
    ret = w.T @ er
    risk = cp.quad_form(w, cov)
    constraints = [
        cp.sum(w) == 1,
        w <= 0.1,
        w >= 0,
        ]
    obj = cp.Maximize(ret - gamma*risk)
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return w.value

def get_MV_weights(cov):
    n = cov.shape[0]
    w = cp.Variable((n, 1))
    risk = cp.quad_form(w, cov)
    constraints = [
        cp.sum(w) == 1,
        w <= 0.1,
        w >= 0,
        ]
    obj = cp.Minimize(risk)
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return w.value

def get_weights(start, ete_mats):
    ete_out = ete_mats[start].sum(axis=0).values.reshape(-1, 1)
    n = len(ete_out)
    w = cp.Variable((n, 1))
    obj = cp.Minimize(w.T @ ete_out)
    constraints = [
        cp.sum(w) == 1,
        w <= 0.1,
        w >= 0,
        ]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return w.value

# %%

mod = TransferEntropy(assets=assets)
mo_df = mod.prices.resample('M').last()
mo_ret_df = (mo_df/mo_df.shift(1).values-1).dropna()
EXP_RETS = mo_ret_df.mean().values.reshape(-1, 1)

e_perf = []
e_perf_ete = []
mv_perf = []
mv_perf_ete = []
capm = defaultdict(list)
capm_ete = defaultdict(list)

gammas = [0.1, 1, 10]
with tqdm(total=iters) as pbar:
    for start, end in zip(months[:-1], months[1:]):
        end -= timedelta(1)
        mod.set_timeperiod(start, end)

        # get month's returns, cov, and ete matrices
        cov = np.cov(mod.data.values, rowvar=False)
        ete_out = ete_mats[start].sum(axis=0).values.reshape(-1, 1)
        r = (mod.prices.iloc[-1]/mod.prices.iloc[0]-1).values

        # get strategy weights
        we = get_weights(start, ete_mats)
        wmv = get_MV_weights(cov)

        e_perf.append(np.squeeze(we.T @ r))
        e_perf_ete.append(np.squeeze(we.T @ ete_out))

        mv_perf.append(np.squeeze(wmv.T @ r))
        mv_perf_ete.append(np.squeeze(wmv.T @ ete_out))

        for gamma in gammas:
            w_capm = get_CAPM_weights(EXP_RETS, cov, gamma)
            capm[gamma].append(np.squeeze(w_capm.T @ r))
            capm_ete[gamma].append(np.squeeze(w_capm.T @ ete_out))

        pbar.update(1)

# %%
alpha=0.75
lw=2
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
cmap2 = sns.color_palette('Reds_r', n_colors=len(gammas)*2)
ax.plot(months, cum_rets(e_perf), alpha=alpha,
    label='ETE', lw=lw, c='steelblue')

ax.plot(months, cum_rets(mv_perf), alpha=alpha,
    label='MV', lw=lw, c='forestgreen')
for i, gamma in enumerate(reversed(gammas[1:])):
    ax.plot(months, cum_rets(capm[gamma]), alpha=alpha,
        label=f'CAPM $\\gamma={gamma}$', lw=lw, c=cmap2[i])

fig.autofmt_xdate()
plt.ylabel('Cumulative Return')
plt.xlabel('Time')
plt.legend()
plt.tight_layout()
eu.save_fig(f'../plots/portfolio_comparison.png', dpi=300)
plt.show()
eu.latex_figure(f'../plots/portfolio_comparison')
# %%
tbl = pd.DataFrame({
        'ETE': e_perf,
        'MV': mv_perf,
        'CAPM 1': capm[1],
        'CAPM 10': capm[10],
        }, index=months[1:])

tbl = (tbl.mean()*12).to_frame().join((tbl.std()*np.sqrt(12)).to_frame(),
        rsuffix='vol')

tbl.columns = 'Return Volatility'.split()
tbl['Sharpe'] = tbl['Return']/tbl['Volatility']
tbl['Return'] *= 100
tbl['Volatility'] *= 100

tbl2 = pd.DataFrame({
        'ETE': e_perf_ete,
        'MV': mv_perf_ete,
        'CAPM 1': capm_ete[1],
        'CAPM 10': capm_ete[10],
        }, index=months[1:])
tbl = tbl.join(tbl2.mean().to_frame())
tbl.columns = 'Return Volatility Sharpe ETE'.split()
eu.latex_print(tbl, prec=2)
