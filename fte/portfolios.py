from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
from tqdm import tqdm

from transfer_entropy import TransferEntropy
# %%
%matplotlib inline
# %%
bonddf = pd.read_csv('../data/fixed_income.csv',
            usecols=['Dates', 'IEF'],
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True)

stockdf = pd.read_csv('../data/equities.csv',
            usecols=['Dates', 'SPY'],
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True)

df = stockdf.join(bonddf)
# %%
start = '1/1/2010'
end = '12/31/2018'
test_df = df.loc[start:end]

# %%
self = TransferEntropy(data=test_df)

SPY_rets = test_df.SPY.values[1:]/test_df.SPY.values[:-1]-1
IEF_rets = test_df.IEF.values[1:]/test_df.SPY.values[:-1]-1

dates = test_df.index
start, end = dates[0], dates[-1]
# %%
w = 90
step = 5

corrs = []
IEF_SPYs = []
SPY_IEFs = []
IEF_SPYs_eff = []
SPY_IEFs_eff = []
for win, date in tqdm(zip(range(0, len(test_df)-w, step), dates[w::step])):
    x = SPY_rets[win:win+w]
    y = IEF_rets[win:win+w]
    corrs.append(np.corrcoef(x, y)[1, 0])

    ts = dates[win]
    self.set_timeperiod(ts, date)
    self.compute_transfer_entropy()
    IEF_SPYs.append(self.te.iloc[0, 1])
    SPY_IEFs.append(self.te.iloc[1, 0])

    self.compute_effective_transfer_entropy(sims=25)
    IEF_SPYs_eff.append(self.ete.iloc[0, 1])
    SPY_IEFs_eff.append(self.ete.iloc[1, 0])

# %%

lw = 1
alpha = 0.7
pad = 0.5
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].plot(dates[w::step], corrs,
    lw=lw, alpha=alpha, c='steelblue',
    label='Correlation')
axes[0].set_ylabel('Pearson Correlation', labelpad=pad)
axes[0].get_yaxis().set_label_coords(-0.05, 0.5)
# axes[0].set_ylim(-.5, 0.7)

axes[1].plot(dates[w::step], IEF_SPYs, '-',
    lw=lw, alpha=alpha, c='darkred',
    label='IEF$\\rightarrow$SPY')
axes[1].plot(dates[w::step], SPY_IEFs, '-',
    lw=lw, alpha=alpha, c='black',
    label='SPY$\\rightarrow$IEF')
axes[1].set_ylabel('Transfer Entropy', labelpad=pad)
axes[1].get_yaxis().set_label_coords(-0.05, 0.5)
# axes[1].set_ylim(-0.1, 0.7)
# axes[1].yaxis.label.set_color('black')

axes[2].plot(dates[w::step], IEF_SPYs_eff, '-',
    lw=lw, alpha=alpha, c='darkred',
    label='IEF$\\rightarrow$SPY')
axes[2].plot(dates[w::step], SPY_IEFs_eff, '-',
    lw=lw, alpha=alpha, c='black',
    label='SPY$\\rightarrow$IEF')
axes[2].set_ylabel('Effective Transfer Entropy', labelpad=pad)
axes[2].get_yaxis().set_label_coords(-0.05, 0.5)

axes[0].grid(False)
axes[1].grid(False)
axes[2].grid(False)

axes[1].legend(loc='upper left')
axes[2].legend(loc='upper left')
# axes[0].legend(loc='upper left')

fig.autofmt_xdate()

plt.xlabel('Time')
fig.tight_layout()
plt.show()
# %%
lw = 1
alpha = 0.7
pad = 0.5
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
axes[0].plot(dates[w::step], corrs,
    lw=lw, alpha=alpha, c='steelblue',
    label='Correlation')
axes[0].set_ylabel('Pearson Correlation', labelpad=pad)
axes[0].get_yaxis().set_label_coords(-0.05, 0.5)
# axes[0].set_ylim(-.5, 0.7)

axes[1].plot(dates[w::step],
    np.mean([IEF_SPYs, SPY_IEFs], axis=0), '-',
    lw=lw, alpha=alpha, c='darkred')

axes[1].set_ylabel('Transfer Entropy', labelpad=pad)
axes[1].get_yaxis().set_label_coords(-0.05, 0.5)
# axes[1].set_ylim(-0.1, 0.7)
# axes[1].yaxis.label.set_color('black')

axes[2].plot(dates[w::step],
    np.mean([IEF_SPYs_eff, SPY_IEFs_eff], axis=0), '-',
    lw=lw, alpha=alpha, c='darkred')

axes[2].set_ylabel('Effective Transfer Entropy', labelpad=pad)
axes[2].get_yaxis().set_label_coords(-0.05, 0.5)

axes[0].grid(False)
axes[1].grid(False)
axes[2].grid(False)

# axes[1].legend(loc='upper left')
# axes[2].legend(loc='upper left')
# axes[0].legend(loc='upper left')

fig.autofmt_xdate()

plt.xlabel('Time')
fig.tight_layout()
plt.show()

# %%
def get_portfolio_info(self, metric='TE', freq='M'):
    # get full time period start / ends
    full_days = self.prices.index
    full_start, full_end = full_days[0], full_days[-1]
    periods = self.prices.index.to_period(freq).unique()

    times = []
    weights = defaultdict(list)
    returns = defaultdict(list)
    stdevs = defaultdict(list)

    for i, period in enumerate(periods[:-1]):
        # set time period for current month
        days = self._prices.loc[str(period)].index
        start, end = days[0], days[-1]
        self.set_timeperiod(start, end)

        if metric == 'TE':
            self.compute_transfer_entropy()
            mat = self.te.copy()
        else:
            self.compute_effective_transfer_entropy()
            mat = self.ete.copy()

        out_sums, in_sums = mat.sum(axis=0), mat.sum(axis=1)
        w_out = out_sums.values/out_sums.sum()
        w_in = in_sums.values/in_sums.sum()

        # get returns and cov matrix for next time period
        days = self._prices.loc[str(periods[i+1])].index
        start, end = days[0], days[-1]
        self.set_timeperiod(start, end)

        rets = (self.prices.iloc[-1]/self.prices.iloc[0]-1).values
        cov = self.data.cov().values

        # store information
        times.append(str(periods[i]))

        weights['in'].append(w_in)
        returns['in'].append(w_in@rets)
        stdevs['in'].append(np.sqrt(w_in.T@cov@w_in))

        weights['out'].append(w_out)
        returns['out'].append(w_out@rets)
        stdevs['out'].append(np.sqrt(w_out.T@cov@w_out))

        w_equal = np.ones(len(w_out))/len(w_out)
        weights['equal'].append(w_equal)
        returns['equal'].append(w_equal@rets)
        stdevs['equal'].append(np.sqrt(w_equal.T@cov@w_equal))


    times.append(str(periods[i+1]))
    self.set_timeperiod(full_start, full_end)
    return times, weights, returns, stdevs

def plot_cumulative_returns(self, times: list, rets: dict):
    methods = rets.keys()
    cum_rets = defaultdict(list)
    for method in methods:
        cum_rets[method].append(1)
        for i, ret in enumerate(rets[method]):
                cum_rets[method].append(cum_rets[method][i]*(1+ret))

    plt.figure(figsize=(10, 6))
    cmap = sns.color_palette('Blues', n_colors=len(methods))
    for method, c in zip(methods, cmap):
        plt.plot(
            pd.to_datetime(times),
            cum_rets[method],
            lw=2,
            label=method,
            c=c)
    plt.xticks(rotation=45)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%

self = TransferEntropy()
start, end = '1/1/2008', '12/31/2009'
self.set_timeperiod(start, end)
# %%
times, ws, rets, stds = get_portfolio_info(self)
# %%
plot_cumulative_returns(self, times, rets)
