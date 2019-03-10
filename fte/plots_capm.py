from datetime import timedelta

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
# fx = 'FXA FXB FXC FXE FXF FXY'.split()
assets = eqs + fi + cmdtys
# %%
ete_mats = pd.read_csv('../ete_Q.csv', index_col=[0, 1], parse_dates=True,
    infer_datetime_format=True)
ete_mats = ete_mats[assets].copy()

def simulations(start, end, ete_mats, sims=20000, assets=assets):
    mod = TransferEntropy(assets=assets)
    mod.set_timeperiod(start, end)
    mod.compute_effective_transfer_entropy(sims=30, pbar=True)
    te_mat = mod.te.values.copy()
    ete_mat = mod.ete.values.copy()
    # ete_mat = ete_mats.loc[start]
    # ete_mat = ete_mat.T[assets].T.values.copy()
    ete_out = ete_mat.sum(axis=0).reshape(-1, 1)
    ete_in = ete_mat.sum(axis=1).reshape(-1, 1)


    df = mod.prices.copy()
    df = df[assets]
    prices = df.loc[start:end-timedelta(1)]
    returns = (prices/prices.shift(1).values-1).dropna()
    cov = np.cov(returns, rowvar=False)
    rets = returns.mean().values.reshape(-1, 1)
    # %%

    n = cov.shape[0]

    ete_outs = np.zeros(sims)
    ete_ins = np.zeros(sims)
    ete_netout = np.zeros(sims)
    ete_total = np.zeros(sims)
    # ete_quad = np.zeros(sims)
    port_vols = np.zeros(sims)
    port_rets = np.zeros(sims)

    for sim in range(sims):
        w = np.random.uniform(-0.5, 0.5, size=(n, 1))
        w /= w.sum()

        r = np.squeeze(w.T @ rets)*4
        vol = np.sqrt(np.squeeze(w.T @ cov @ w)*4)

        if vol <= .2:
            port_vols[sim] = vol
            port_rets[sim] = r

            ete_outs[sim] = np.squeeze(w.T @ ete_out)
            ete_ins[sim] = np.squeeze(w.T @ ete_in)
            ete_netout[sim] = ete_outs[sim] - ete_ins[sim]
            ete_total[sim] = ete_outs[sim] + ete_ins[sim]
            # ete_quad[sim] = np.squeeze(w.T @ ete_mat @ w)

    sr = port_rets/port_vols
    mask = np.argwhere(port_vols!=0)
    ete_outs = ete_outs[mask]
    ete_ins = ete_ins[mask]
    ete_netout = ete_netout[mask]
    ete_total = ete_total[mask]
    # ete_quad = ete_quad[mask]
    port_vols = port_vols[mask]
    port_rets = port_rets[mask]

    return [port_rets, port_vols, ete_total, ete_netout, ete_outs, ete_ins, sr]

# %%
mod = TransferEntropy(assets=assets)
months = mod.prices.index.to_period('Q').to_timestamp().unique()

iters = len(months) - 1
with tqdm(total=iters) as pbar:
    for start, end in tqdm(zip(months[:-1], months[1:])):
        start, end = months[0], months[-1]

        output = simulations(start=start, end=end, ete_mats=None,
                                sims=20000, assets=assets)


        port_rets, port_vols, ete_total, ete_netout, ete_outs, ete_ins, sr = output

        start = str(start.to_period('Q'))
        end  = str(end.to_period('Q'))
        # %%
        # plt.figure(figsize=(10, 10))
        # sns.distplot(ete_total, bins=30, rug=True)
        # plt.xlabel('Effective Transfer Entropy')
        # plt.tight_layout()
        # eu.save_fig(f'../plots/ete_pdf_{start}_{end}', dpi=300)
        # %%
        visdf = pd.DataFrame({'Return': port_rets.flatten(),
            'Volatility': port_vols.flatten(),
            'ETE In': ete_ins.flatten(),
            'ETE Out': ete_outs.flatten(),
            # 'Net Out': ete_netout.flatten(),
            # 'Total ETE': ete_total.flatten(),
            }, index=np.arange(len(ete_total)))
        # visdf.to_csv('../visdf.csv')
        # %%
        g = sns.pairplot(visdf, diag_kind='kde', aspect=1.4)
        g.map_diag(plt.hist, bins=20, color='steelblue')
        # g.map_offdiag(sns.kdeplot, n_levels=3, linewidths=2)
        eu.save_fig(f'../plots/pairplot_{start}_{end}', dpi=300)
        # %%
        # plt.figure(figsize=(10, 6))
        # plt.scatter(port_vols, 1+port_rets,
        #     c=ete_total, cmap='hot_r',
        #     alpha=0.75, s=8)
        # cbar = plt.colorbar()
        # cbar.ax.set_title('ETE')
        # plt.xlim(None, 0.1)
        # plt.xlabel('Volatility')
        # plt.ylabel('Return')
        # plt.tight_layout()
        # eu.save_fig(f'../plots/capm_frontier_{start}_{end}', dpi=300)
        # %%
        # mets = [ete_ins, ete_total, ete_outs, ete_netout]
        # lbls = '$In$ $In+Out$ $Out$ $Out-In$'.split()
        # fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey=True)
        # i = 0
        # for ax, met, lbl in zip(axes.flatten(), mets, lbls):
        #     im = ax.scatter(met, port_rets,
        #             c=port_vols, cmap='hot_r',
        #             alpha=0.75, s=8)
        #     if i % 2 == 0: # y label only on left column
        #         ax.set_ylabel('Return')
        #     ax.set_xlabel(lbl)
        #     i += 1
        # plt.tight_layout()
        # fig.subplots_adjust(right=.8)
        # cbar = fig.add_axes([0.85, 0.1, 0.05 , 0.8])
        # cbar.set_title('Vol', fontsize=12)
        # fig.colorbar(im, cax=cbar)
        # eu.save_fig(f'../plots/test_entropies_returns_{start}_{end}', dpi=300)
        # %%
        # mets = [ete_ins, ete_total, ete_outs, ete_netout]
        # lbls = '$In$ $In+Out$ $Out$ $Out-In$'.split()
        # fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)
        # i = 0
        # for ax, met, lbl in zip(axes.flatten(), mets, lbls):
        #     ax.scatter(port_vols, met,
        #         c=port_rets, cmap='hot_r',
        #         alpha=0.75, s=8)
        #     # cbar = ax.colorbar()
        #     # cbar.ax.set_title('Sharpe')
        #     if i >= 2:
        #         ax.set_xlabel('Volatility')
        #     ax.set_ylabel(lbl)
        #     i += 1
        #     # plt.ylabel('Return')
        # plt.tight_layout()
        # fig.subplots_adjust(right=.8)
        # cbar = fig.add_axes([0.85, 0.1, 0.05 , 0.8])
        # cbar.set_title('Return', fontsize=12)
        # fig.colorbar(im, cax=cbar)
        # eu.save_fig(f'../plots/test_vols_entropies_{start}_{end}', dpi=300)
        pbar.update(1)
# %%
