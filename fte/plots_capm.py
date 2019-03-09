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

# %%
def simulations(start, end, sims=20000, assets=assets):
    mod = TransferEntropy(assets=assets)
    mod.set_timeperiod(start, end)
    mod.compute_effective_transfer_entropy(sims=50, pbar=False)
    te_mat = mod.te.values.copy()
    ete_mat = mod.ete.values.copy()
    ete_out = ete_mat.sum(axis=0).reshape(-1, 1)
    ete_in = ete_mat.sum(axis=1).reshape(-1, 1)

    df = mod.prices.copy()
    prices = df.loc[start:end].resample('M').last()
    returns = (prices/prices.shift(1).values-1).dropna()
    cov = np.cov(returns, rowvar=False)
    rets = returns.mean().values.reshape(-1, 1)
    # %%

    n = cov.shape[0]

    ete_outs = np.zeros(sims)
    ete_ins = np.zeros(sims)
    ete_netout = np.zeros(sims)
    ete_netin = np.zeros(sims)
    ete_total = np.zeros(sims)
    # ete_quad = np.zeros(sims)
    port_vols = np.zeros(sims)
    port_rets = np.zeros(sims)

    for sim in range(sims):
        w = np.random.uniform(-0.5, 0.5, size=(n, 1))
        w /= w.sum()

        r = np.squeeze(w.T @ rets)*12
        vol = np.sqrt(np.squeeze(w.T @ cov @ w)*12)

        if vol <= .2:
            port_vols[sim] = vol
            port_rets[sim] = r

            ete_outs[sim] = np.squeeze(w.T @ ete_out)
            ete_ins[sim] = np.squeeze(w.T @ ete_in)
            ete_netin[sim] = ete_ins[sim] - ete_outs[sim]
            ete_netout[sim] = ete_outs[sim] - ete_ins[sim]
            ete_total[sim] = ete_outs[sim] + ete_ins[sim]
            # ete_quad[sim] = np.squeeze(w.T @ ete_mat @ w)

    sr = port_rets/port_vols
    mask = np.argwhere(port_vols!=0)
    ete_outs = ete_outs[mask]
    ete_ins = ete_ins[mask]
    ete_netout = ete_netout[mask]
    ete_netin = ete_netin[mask]
    ete_total = ete_total[mask]
    # ete_quad = ete_quad[mask]
    port_vols = port_vols[mask]
    port_rets = port_rets[mask]

    return [port_rets, port_vols, ete_total, ete_netout, ete_netin,
            ete_outs, ete_ins, sr]

# %%
years = ['2010-01-01', '2011-01-01', '2012-01-01', '2013-01-01', '2014-01-01',
        '2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01']

iters = len(years) - 1
with tqdm(total=iters) as pbar:
    for start, end in tqdm(zip(years[:-1], years[1:])):
    # for start, end in zip(['2010-01-01'], ['2019-04-15']):
        output = simulations(start=start, end=end,
                                sims=20000, assets=assets)

        port_rets, port_vols, ete_total, ete_netout, ete_netin, ete_outs, ete_ins, sr = output
        # %%
        plt.figure(figsize=(10, 10))
        sns.distplot(ete_total, bins=30, rug=True)
        plt.xlabel('Effective Transfer Entropy')
        plt.tight_layout()
        eu.save_fig(f'../plots/ete_pdf_{start}_{end}', dpi=300)
        # %%
        visdf = pd.DataFrame({'Return': port_rets.flatten(),
            'Volatility': port_vols.flatten(),
            'ETE': ete_total.flatten()}, index=np.arange(len(ete_total)))

        g = sns.pairplot(visdf, diag_kind='kde')
        g.map_diag(plt.hist, bins=30, color='steelblue')
        g.map_offdiag(sns.kdeplot, n_levels=4, color='steelblue', linewidths=2)
        eu.save_fig(f'../plots/pairplot_{start}_{end}', dpi=300)
        # %%
        plt.figure(figsize=(10, 6))
        plt.scatter(port_vols, 1+port_rets,
            c=ete_total, cmap='hot_r',
            alpha=0.75, s=8)
        cbar = plt.colorbar()
        cbar.ax.set_title('ETE')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.tight_layout()
        eu.save_fig(f'../plots/capm_frontier_{start}_{end}', dpi=300)
        # %%
        mets = [ete_ins, ete_netin, ete_outs, ete_netout]
        lbls = '$In$ $In-Out$ $Out$ $Out-In$'.split()
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey=True)
        i = 0
        for ax, met, lbl in zip(axes.flatten(), mets, lbls):
            im = ax.scatter(met, port_rets,
                    c=port_vols, cmap='hot_r',
                    alpha=0.75, s=8)
            if i % 2 == 0: # y label only on left column
                ax.set_ylabel('Return')
            ax.set_xlabel(lbl)
            i += 1
        plt.tight_layout()
        fig.subplots_adjust(right=.8)
        cbar = fig.add_axes([0.85, 0.1, 0.05 , 0.8])
        cbar.set_title('Vol', fontsize=12)
        fig.colorbar(im, cax=cbar)
        eu.save_fig(f'../plots/test_entropies_returns_{start}_{end}', dpi=300)
        # %%
        mets = [ete_ins, ete_netin, ete_outs, ete_netout]
        lbls = '$In$ $In-Out$ $Out$ $Out-In$'.split()
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
        i = 0
        for ax, met, lbl in zip(axes.flatten(), mets, lbls):
            ax.scatter(port_vols, met,
                c=port_rets, cmap='hot_r',
                alpha=0.75, s=8)
            # cbar = ax.colorbar()
            # cbar.ax.set_title('Sharpe')
            if i >= 2:
                ax.set_xlabel('Volatility')
            ax.set_ylabel(lbl)
            i += 1
            # plt.ylabel('Return')
        plt.tight_layout()
        fig.subplots_adjust(right=.8)
        cbar = fig.add_axes([0.85, 0.1, 0.05 , 0.8])
        cbar.set_title('Return', fontsize=12)
        fig.colorbar(im, cax=cbar)
        eu.save_fig(f'../plots/test_vols_entropies_{start}_{end}', dpi=300)
        pbar.update(1)
# %%
