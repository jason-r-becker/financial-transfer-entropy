import json
from collections import defaultdict
from glob import glob

import empiricalutilities as eu
import numpy as np
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import pandas as pd

from transfer_entropy import TransferEntropy

plt.style.use('fivethirtyeight')
%matplotlib qt

eqs = 'SPY DIA XLK XLV XLF IYZ XLY XLP XLI XLE XLU XME IYR XLB XPH IWM PHO ' \
    'SOXX WOOD FDN GNR IBB ILF ITA IYT KIE PBW ' \
    'AFK EZA ECH EWW EWC EWZ EEM EIDO EPOL EPP EWA EWD EWG EWH EWJ EWI EWK ' \
    'EWL EWM EWP EWQ EWS EWT EWU EWY GXC HAO EZU RSX TUR'.split()
fi = 'AGG SHY IEI IEF TLT TIP LQD HYG MBB'.split()
cmdtys = 'GLD SLV DBA DBC USO UNG'.split()
fx = 'FXA FXB FXC FXE FXF FXS FXY'.split()
assets = eqs + fi + cmdtys + fx

# plot_temporal_vol_and_matrices(assets)
# make_appendix_table(assets)

# %%


def plot_temporal_vol_and_matrices(assets):
    mod = TransferEntropy(assets=assets)

    periods = [
        ('7/1/2011', '8/1/2011'),
        ('8/1/2011', '9/1/2011'),
        ('9/1/2011', '10/1/2011'),
        ('10/1/2011', '11/1/2011'),
        ('11/1/2011', '12/1/2011'),
        ('12/1/2011', '1/1/2012'),
        ('1/1/2012', '2/1/2012'),
        ]
    dates = [pd.to_datetime(p[0]) for p in periods]
    xlabs = [date.strftime('%b %Y') for date in dates]

        
    df = mod.data
    df = df**2
    df = df.resample('M').sum()
    df = (df.mean(axis=1) * 12)**0.5
    df = df[(df.index >= pd.to_datetime('6/1/2011'))
            & (df.index <= pd.to_datetime(periods[-1][-1]))]


    fig, ax = plt.subplots(1, 1, figsize=(20, 3))
    ax.plot(df, c='steelblue')
    tick = mtick.StrMethodFormatter('{x:.0%}')
    ax.yaxis.set_major_formatter(tick)
    ax.set_xlabel('Date')
    ax.set_ylabel('Annualized Volatility')
    eu.save_fig('temporal_vol', dir='../figures')
    plt.show()

    # %%


    # Find maximum transfer entropy for all periods.
    te_max, ete_max, ete_min = 0, 0, 0

    for i, period in enumerate(periods):
        mod.set_timeperiod(*period)
        mod.compute_transfer_entropy()
        te_max = max(np.max(mod.te.values), te_max)
        mod.compute_effective_transfer_entropy(sims=3)
        ete_max = max(np.max(mod.ete.values), ete_max)
        ete_min = min(np.min(mod.ete.values), ete_min)
        
    # %%
    fig, axes = plt.subplots(3, len(periods), figsize=(20, 10),
                             sharex=True, sharey=True)



    for i, period in enumerate(periods):
        mod.set_timeperiod(*period)
        mod.plot_corr(method='pearson', ax=axes[0, i],
                      cbar=False, labels=False)
        mod.compute_transfer_entropy()
        mod.plot_te(ax=axes[1, i], cbar=False, labels=False,
                    vmin=0, vmax=te_max)
        mod.compute_effective_transfer_entropy(sims=5)
        mod.plot_te(ax=axes[2, i], te='ete', cbar=False, labels=False,
                    vmin=ete_min, vmax=0.75*te_max)

        axes[2, i].set_xlabel(xlabs[i])


    axes[0, 0].set_ylabel('Correlation')
    axes[1, 0].set_ylabel('Transfer Entropy')
    axes[2, 0].set_ylabel('Effective Transfer Entropy')
    plt.tight_layout()
    # eu.save_fig('temporal_matrices', dir='../figures')
    plt.show()



def make_appendix_table(assets=None):
    """Build asset information table for appendix."""
    # Read all data.
    def read_file(fid):
        """Read data file as DataFrame."""
        df = pd.read_csv(
            fid,
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
            )
        sorted_cols = sorted(list(df))
        sorted_cols
        return df[sorted_cols]
        
    fids = glob('../data/*.csv')
    df = pd.DataFrame().join([read_file(fid) for fid in fids], how='outer')
    
    # Get asset class for each asset.
    asset_class_map = {}
    for fid in fids:
        asset_class = {
            'fx': 'FX',
            'commodities': 'Commodity',
            'alternatives': 'Alternative',
            'equities': 'Equity',
            'fixed_income': 'Fixed Income',
            }[fid.split('.csv')[0].rsplit('/', 1)[1]]
        temp_df = read_file(fid)
        for asset in temp_df.columns:
            asset_class_map[asset] = asset_class
    
    # Subset data to specified assets if provided.
    if assets is not None:
        df = df[assets].copy()
    tickers = list(df)
    
    # Read name conversions dict.
    with open('../data/asset_mapping.json', 'r') as fid:
        name_map = json.load(fid)
    
    table_data = defaultdict(list)
    for ticker in tickers:
        dates = df[ticker].dropna().index
        table_data['Ticker'].append(ticker)
        table_data['Security Name'].append(name_map[ticker])
        table_data['Asset Class'].append(asset_class_map[ticker])
        table_data['Start Date'].append(dates[0].strftime('%m/%d/%Y'))
        table_data['End Date'].append(dates[-1].strftime('%m/%d/%Y'))
        
    table = pd.DataFrame(table_data)
    table.loc[table['Asset Class']=='Alternative']
    cap = 'Summary of studied assets.'
    col_fmt = 'lllccc'
    eu.latex_print(table, hide_index=True, col_fmt=col_fmt, caption=cap,
                   adjust=True, greeks=False)

def plot_networks(assets):
    # %%
    start = '1/1/2011'
    end = '12/31/2018'
    sims = 50
    
    mod = TransferEntropy(assets=assets)
    mod.set_timeperiod(start, end)
    mod.compute_effective_transfer_entropy(sims=sims, pbar=True)
    
    # %%
    # Settings.
    corr_thresh = 0.65
    ete_thresh_high = 0.0085
    ete_thresh_low = 0.006
    # ete_thresh = (ete_thresh_high + ete_thresh_low) / 2
    ete_thresh = ete_thresh_low
    
    # Plot correlation matrix.
    mod.plot_corr(cbar=True)
    plt.title('Correlation')
    plt.show()

    # Find network graph coordinates.
    fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    fig.suptitle('Correlation Magnitude')
    mod.plot_corr_network(
        network_type='circle',
        threshold=corr_thresh,
        ax=axes[0],
        )
    mod.plot_corr_network(
        network_type='cluster',
        threshold=corr_thresh,
        ax=axes[1],
        )
    plt.show()

    # Plot effective transfer entropy.
    mod.plot_te(vmax=0.5*np.max(mod.te.values))
    plt.title('Effective Transfer Entropy')
    plt.show()

    # Plot effective transfer entropy out.
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    fig.suptitle('Transfer Entropy Out')
    mod.plot_ete_network(
        network_type='circle',
        threshold=ete_thresh,
        ax=axes[0],
        )
    mod.plot_ete_network(
        network_type='cluster',
        threshold=ete_thresh_low,
        ax=axes[1],
        )
    axes[1].set_title('Low Threshold')
    mod.plot_ete_network(
        network_type='cluster',
        threshold=ete_thresh_high,
        ax=axes[2],
        )
    axes[2].set_title('High Threshold')
    plt.show()

    # Plot effective transfer entropy in.
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    fig.suptitle('Transfer Entropy In')
    mod.plot_ete_network(
        network_type='circle',
        node_value='in',
        threshold=ete_thresh,
        cmap='Purples',
        ax=axes[0],
        )
    mod.plot_ete_network(
        network_type='cluster',
        node_value='in',
        threshold=ete_thresh_low,
        cmap='Purples',
        ax=axes[1])
    axes[1].set_title('Low Threshold')
    mod.plot_ete_network(
        network_type='cluster',
        node_value='in',
        threshold=ete_thresh_high,
        cmap='Purples',
        ax=axes[2])
    axes[2].set_title('High Threshold')
    plt.show()

    # %%
def plot_randomized_transfer_entropy_hists():
    n = len(self.assets)
    fig, axes = plt.subplots(n, n, figsize=(20, 20), sharex=True, sharey=True)
    for i in range(n):
        for j in range(n):
            te = self.te.iloc[i, j]
            rte_array = self.rte_tensor[i, j, :]
            z = norm(np.mean(rte_array), np.std(rte_array)).cdf(te)
            if z < cutoff:
                self.ete[i, j] = 0
                c = 'firebrick'
            else:
                c = 'darkgreen'
            
            axes[i, j].hist(self.rte_tensor[i, j, :], bins=30, density=True,
                            alpha=0.7, color='steelblue', label='_nolegend_')
            axes[i, j].axvline(te, lw=2, color=c,
                label=f'z: {z:.2f}')
            axes[i, j].legend()
    plt.show()
