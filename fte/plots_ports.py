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
eqs = 'SPY DIA XLK XLV XLF IYZ XLY XLP XLI XLE XLU XME IYR XLB XPH IWM PHO ' \
    'SOXX WOOD FDN GNR IBB ILF ITA IYT KIE PBW ' \
    'AFK EZA ECH EWW EWC EWZ EEM EIDO EPOL EPP EWA EWD EWG EWH EWJ EWI EWK ' \
    'EWL EWM EWP EWQ EWS EWT EWU EWY GXC HAO EZU RSX TUR'.split()
fi = 'AGG SHY IEI IEF TLT TIP LQD HYG MBB'.split()
cmdtys = 'GLD SLV DBA DBC USO UNG'.split()
fx = 'FXA FXB FXC FXE FXF FXY'.split()
assets = eqs + fi + cmdtys + fx

def cum_rets(rets):
    cum_rets = []
    cum_rets.append(1)
    for i, ret in enumerate(rets):
        cum_rets.append(cum_rets[i]*(1+ret))
    return cum_rets

# %%
ete_mats = {}
mod = TransferEntropy(assets=assets)
period = 'Q'
months = mod.prices.index.to_period(period).unique().to_timestamp()
iters = len(months)-24

with tqdm(total=iters) as pbar:
    for start, end in zip(months[:-1], months[1:]):
        end -= timedelta(1)
        mod.set_timeperiod(start, end)
        mod.compute_effective_transfer_entropy(sims=30, bins=6,
                                               std_threshold=1)
        ete = mod.ete.copy()
        ete_mats[start] = ete
        pbar.update(1)

ete_df = pd.concat(ete_mats)
ete_df.to_csv(f'../ete_{period}.csv')
# %%
q = 4
res = defaultdict(dict)

mod = TransferEntropy(assets=assets)
iters = len(months)-1


for start, end in zip(months[:-1], months[1:]):
    ete = ete_mats[start]
    ete_out = ete.sum(axis=0)
    ete_in = ete.sum(axis=1)

    end -= timedelta(1)
    mod.set_timeperiod(start, end)

    returns = mod.prices.iloc[-1]/mod.prices.iloc[0]-1
    vols = mod.data.std()

    names = 'eteout etein etenetout etetotal'.split()
    for name, ETE in zip(names, [ete_out, ete_in,
                                 ete_out-ete_in, ete_in+ete_out]):

        df = pd.DataFrame({'returns': returns, 'vol': vols, name: ETE})
        df['q'] = pd.qcut(ETE, q=q, labels=False)

        res[name][start] = df.groupby('q').agg('mean').reset_index().copy()

# %%
q_rets = {}

for name in names:
    resdf = res[name]
    resdf = pd.concat(resdf)
    resdf.index = resdf.index.droplevel(1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cmap = sns.color_palette('Blues_r', n_colors=4)
    for c, qtile in zip(cmap, range(q)):
        q_rets[qtile] = resdf[resdf['q']==qtile]['returns'].values
        ax.plot(months, cum_rets(q_rets[qtile]), c=c,
            lw=2, alpha=1, label=f'Quartile {qtile+1}')

    fig.autofmt_xdate()
    plt.ylabel('Cumulative Return')
    plt.xlabel('Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../plots/{name}_quartile_returns.png', dpi=300)
    eu.latex_figure(f'../data/plots/{name}_quartile_returns.png')
# %%


for name in names:
    table = defaultdict(dict)
    resdf = res[name]
    resdf = pd.concat(resdf)
    resdf.index = resdf.index.droplevel(1)
    table
    for qtile in range(q):
        table[qtile]['r'] = resdf[resdf['q']==qtile]['returns'].mean()*12
        table[qtile]['v'] = resdf[resdf['q']==qtile]['returns'].std()*np.sqrt(12)
        table[qtile][name] = resdf[resdf['q']==qtile][name].mean()

    table = pd.DataFrame.from_dict(table, orient='index')
    table['sr'] = table['r']/table['v']
    table = table.reset_index()
    table = table[['index', 'r', 'v', 'sr', name]]
    cols = 'Quartile Return Volatility Sharpe'.split()
    cols += [name]
    table.columns = cols
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
        ret >= 0.02,
        ]
    obj = cp.Maximize(ret - gamma*risk)
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return w.value

def get_MV_weights(er, cov):

    n = cov.shape[0]
    w = cp.Variable((n, 1))
    ret = w.T @ er
    risk = cp.quad_form(w, cov)
    constraints = [
        cp.sum(w) == 1,
        w <= 0.1,
        w >= 0,
        ret >= 0.02,
        ]
    obj = cp.Minimize(risk)
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return w.value

def get_weights(er, start, ete):
    n = len(ete)
    w = cp.Variable((n, 1))
    ret = w.T @ er
    obj = cp.Minimize(w.T @ (ete))
    constraints = [
        cp.sum(w) == 1,
        w <= 0.1,
        w >= 0,
        ret >= 0.02,
        ]
    prob = cp.Problem(obj, constraints)
    prob.solve()
    return w.value

# %%
ete_mats = pd.read_csv('../ete_Q.csv', index_col=[0, 1], parse_dates=True,
    infer_datetime_format=True)
ete_mats = ete_mats[assets].copy()

mod = TransferEntropy(assets=assets)
mo_df = mod.prices.resample('Q').last()
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

        ete_mat = ete_mats.loc[start]
        ete_mat = ete_mat.T[assets].T.values.copy()
        ete_out = ete_mat.sum(axis=0).reshape(-1, 1)
        ete_in = ete_mat.sum(axis=1).reshape(-1, 1)
        net_out = ete_out - ete_in

        r = (mod.prices.iloc[-1]/mod.prices.iloc[0]-1).values

        # get strategy weights
        we = get_weights(EXP_RETS, start, net_out)
        wmv = get_MV_weights(EXP_RETS, cov)

        e_perf.append(np.squeeze(we.T @ r))
        e_perf_ete.append(np.squeeze(we.T @ net_out))

        mv_perf.append(np.squeeze(wmv.T @ r))
        mv_perf_ete.append(np.squeeze(wmv.T @ net_out))

        for gamma in gammas:
            w_capm = get_CAPM_weights(EXP_RETS, cov, gamma)
            capm[gamma].append(np.squeeze(w_capm.T @ r))
            capm_ete[gamma].append(np.squeeze(w_capm.T @ net_out))

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
eu.save_fig(f'../plots/portfolio_comparison', dpi=300)
plt.show()
eu.latex_figure(f'../plots/portfolio_comparison')
# %%
tbl = pd.DataFrame({
        'ETE': e_perf,
        'MV': mv_perf,
        'CAPM 1': capm[1],
        'CAPM 10': capm[10],
        }, index=months[1:])

tbl = (tbl.mean()*4).to_frame().join((tbl.std()*np.sqrt(4)).to_frame(),
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
