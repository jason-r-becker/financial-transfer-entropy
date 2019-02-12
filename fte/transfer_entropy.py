import json
import warnings
from collections import Counter
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

plt.style.use('fivethirtyeight')



# %matplotlib qt
# %%

class TransferEntropy:
    """
    Class to compute asset graphs using transfer entropy.
    
    Parameters
    ----------
    data: pd.DataFrame, default=None
        DataFrame of asset log returns with datetime index and no missing data.
        If None, price datafiles are loaded from data directory.
    
    Attributes
    ----------
    data: pd.DataFrame
        DataFrame of asset log returns with datetime index.
    assets: list(str)
        List of asset names.
    corr: pd.DataFrame
        Spearman correlation between assets.
    
    Methods
    -------
    set_timeperiod(start, end): Set starting and ending dates for analysis.
    build_asset_graph(solver): Find graph coordinates.
    compute_transfer_entorpy(bins): Return transfer entropy.
    compute_effective_transfer_entropy: Return effective transfer entropy.
    plot_asset_graph(threshold): Plot asset graph edges that meet threshold.
    plot_corr(method): Plot correlations between assets.
    plot_te(te): Plot transfer entropy heatmap.
    """
    
    def __init__(self, data=None):
        self._prices = data if data is not None else self._load_data()
        self.set_timeperiod('1/1/2014', '1/1/2019')
        
    def _read_file(self, fid):
        """Read data file as DataFrame."""
        df = pd.read_csv(
            fid,
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
            )
        return df.dropna(axis=1)
        
    def _load_data(self):
        """Load data from data directory into single into DataFrame."""
        fids = glob('../data/*.csv')
        df = pd.DataFrame().join(
            [self._read_file(fid) for fid in fids], how='outer')
        return df

    def set_timeperiod(self, start=None, end=None):
        """
        Updata self.data with start and end dates for analysis.
        
        Parameters
        ----------
        start: str or datetime object, default=None
            Starting date for analysis.
        end: str or datetime object, default=None
            Ending date for analysis.
        
        """
        data = self._prices.copy()
        
        # Ignore warnings for missing data.
        warnings.filterwarnings('ignore')
        
        # Subset data by time period.
        if start is not None:
            data = data[data.index >= pd.to_datetime(start)].copy()
        if end is not None:
            data = data[data.index <= pd.to_datetime(end)].copy()
        
        # Drop Sundays.
        keep_ix = [ix.weekday() != 6 for ix in list(data.index)]
        data = data[keep_ix].copy()
        self.data = np.log(data[1:] / data[:-1].values)  # log returns
        self.data.dropna(axis=1, inplace=True)  # Drop assets with missing data.
        
        # Map asset names to DataFrame.
        with open('../data/asset_mapping.json', 'r') as fid:
            asset_map = json.load(fid)
        self.assets = [asset_map.get(a, a) for a in list(self.data)]
        self._n = len(self.assets)
        
        # Rename DataFrame with asset names and init data matrix.
        self.data.columns = self.assets
        self._data_mat = self.data.values
        
    def _euclidean_distance(self, x, i, j):
        """Euclidean distance between points in x-coordinates."""
        m = x.shape[1]
        return sum((x[i, a] - x[j, a])**2 for a in range(m))**0.5

    def _stress_function(self, x):
        """Stress function for Classical Multidimensional Scaling."""
        # Map 1-D input coordinates to 2-D
        x = np.reshape(x, (-1, 2))
        n = x.shape[0]
        
        num, denom = 0, 0
        for i in range(n):
            for j in range(i, n):
                delta = int(i == j)
                euc_d = self._euclidean_distance(x, i, j)
    
                # Build numerator and denominator sums.
                num += (delta - euc_d)**2
                denom += self._distance[i, j]**2
        return (num / denom)**0.5
        
    def build_asset_graph(self, solver, distance=None, verbose=False):
        """
        Build asset graph of transfer entropy with specified threshold.
        
        Parameters
        ----------
        solver: str, default='SLSQP'
            Scipy mimiziation solving technique.
        """
        # Find correlations and distance metric.
        self.corr =  self.data.corr('spearman')
        
        self._distance = distance if distance is not None \
                         else np.sqrt(2 * (1-self.corr.values))
        


        # Solve 2-D coordinate positions.
        def exit_opt(Xi):
            if np.sum(np.isnan(Xi)) > 1:
                raise RuntimeError('Minimize convergence failed.')
                
        def printx(Xi):
            print(Xi)
            exit_opt(Xi)
            
        opt = minimize(
            self._stress_function,
            x0=np.random.rand(2*self._n),
            method=solver,
            tol=1e-3,
            options={'disp': False, 'maxiter': 10000},
            callback=printx if verbose else exit_opt,
            )
        if opt.status != 0:
            raise RuntimeError(opt.message)
        
        self._coordinates = np.reshape(opt.x, (-1, 2))


                
    def plot_asset_graph(self, threshold, all_thresholds=None,
                         ax=None, figsize=(6, 6), fontsize=6):
        """
        Plot asset graph network.
        
        Parameters
        ----------
        threshold: float
            Maximum threshold distance for edges in network.
        all_thresholds: list[float], default=None
            If provided, the colorbar maximum value will be set as the
            maximum threshold, otherwise the given threshold is used.
            This is convenient for keeping a standard scale when plotting
            multiple thresholds.
        ax: matplotlib axis, default=None
            Matplotlib axis to plot figure, if None one is created.
        figsize: list or tuple, default=(6, 6)
            Figure size.
        fontsize: int, default=6
            Fontsize for asset labels.
        """
        # Find edges and nodes.
        edges = {}
        nodes = []
        for i in range(self._n - 1):
            d_i = self._distance[i, :]
            edges_i = np.argwhere(d_i < threshold).reshape(-1)
            edges_i = list(edges_i[edges_i > i])
            edges[i] = edges_i
            if len(edges_i) > 0:
                nodes.append(i)
                nodes.extend(edges_i)
        nodes = list(set(nodes))
        edges = {key: val for key, val in edges.items() if len(val) > 0}
        
        # Store values of edges.
        edge_vals = {}
        for node0, node0_edges in edges.items():
            for node1 in node0_edges:
                edge_vals[(node0, node1)] = self._distance[node0, node1]
                
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            
        x = self._coordinates[:, 0]
        y = self._coordinates[:, 1]
        
        # Get sequential colors for edges based on distance.
        cmap = sns.color_palette('magma', 101).as_hex()
        emin = min(self._distance[self._distance > 0])
        emax = threshold if all_thresholds is None else max(all_thresholds)
        emax += 1e-3
        edge_color = {key: cmap[int(100 * (val-emin) / (emax-emin))]
                      for key, val in edge_vals.items()}

        # Plot edges.
        for node0, node0_edges in edges.items():
            for node1 in node0_edges:
                ix = [node0, node1]
                ax.plot(x[ix], y[ix], c=edge_color[tuple(ix)], lw=2, alpha=0.7)
        
        # Plot asset names over edges.
        box = {'fill': 'white', 'facecolor': 'white', 'edgecolor': 'k'}
        for node in nodes:
            ax.text(x[node], y[node], self.assets[node], fontsize=fontsize,
                    horizontalalignment='center', verticalalignment='center',
                    bbox=box)
    
    def plot_corr(self, method='spearman', ax=None, figsize=(6, 6),
                  fontsize=8):
        """
        Plot correlation of assets.
        
        Parameters
        ----------
        method: {'spearman', 'pearson', 'kendall'}, default='spearman'
            Correlation method.
                - 'spearman': Spearman rank correlation
                - 'pearson': standard correlation coefficient
                - 'kendall': Kendall Tau correlation coefficient
        ax: matplotlib axis, default=None
            Matplotlib axis to plot figure, if None one is created.
        figsize: list or tuple, default=(6, 6)
            Figure size.
        fontsize: int, default=8
            Fontsize for asset labels.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        self._corr = self.data.corr(method)
        sns.heatmap(self._corr, ax=ax, vmin=-1, vmax=1,
                    xticklabels=True, yticklabels=True, cmap='coolwarm')
        plt.setp(ax.get_xticklabels(), fontsize=fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize)
        cbar = ax.collections[0].colorbar
        ticks = np.linspace(-1, 1, 5)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)
    
    def _transfer_entropy_function(self, data, X, Y, bins, shuffle):
        """
        Compute transfer entropy for asset x on lagged x and lagged y
        log returns.
        
        Parameters
        ----------
        X: int
            Index location of asset x, asset to be predicted.
        Y: int
            Index location of asset y, asset which influences asset x.
        bins: int
            Number of bins to place log returns in.
        shuffle: bool
            If True, shuffle all time series for randomized transfer entropy.
        
        Returns
        -------
        te: float
            Transfer entropy of y --> x.
        """
        x = data[1:, X]
        x_lag = data[:-1, X]
        y_lag = data[:-1, Y]
        n = len(x)

        # Find respective historgram bin for each time series value.
        bin_vals = np.reshape(pd.cut(np.concatenate(
            [x, x_lag, y_lag]), bins=bins, labels=False), (-1, 3))
        
        if shuffle:
            for j in range(3):
                np.random.shuffle(bin_vals[:, j])
                
        # Find frequency of occurce for each set of joint vectors.
        p_ilag = Counter(bin_vals[:, 1])
        p_i_ilag = Counter(list(zip(bin_vals[:, 0], bin_vals[:, 1])))
        p_ilag_jlag = Counter(list(zip(bin_vals[:, 1], bin_vals[:, 2])))
        p_i_ilag_jlag = Counter(
            list(zip(bin_vals[:, 0], bin_vals[:, 1], bin_vals[:, 2])))
        
        # Catch warnings as errors for np.log2(0).
        warnings.filterwarnings('error')
        
        # Compute transfer entropy.
        te = 0
        for i in range(bins):
            for ilag in range(bins):
                for jlag in range(bins):
                    try:
                        te_i = (p_i_ilag_jlag.get((i, ilag, jlag), 0) / n
                                * np.log2(p_i_ilag_jlag.get((i, ilag, jlag), 0)
                                          * p_ilag.get(ilag, 0)
                                          / p_i_ilag.get((i, ilag), 0)
                                          / p_ilag_jlag.get((ilag, jlag), 0)
                                          ))
                    except (ZeroDivisionError, RuntimeWarning):
                        te_i = 0
                    te += te_i
        
        # Reset warnings to default.
        warnings.filterwarnings('ignore')
        return te
        
    def compute_transfer_entorpy(self, bins=6, shuffle=False, save=True):
        """
        Compute transfer entropy matrix. Returned matrix is directional
        such that asset on X-axis inluences next day transfer entropy of
        asset on the Y-axis.
        
        Parameters
        ----------
        bins: int, default=6
            Number of bins to place log returns in.
        shuffle: bool, default=False
            If True, shuffle all time series for randomized transfer entropy.
        save: bool, default=True
            If True save result
            
        Returns
        -------
        te: [n x n] nd.array
            Transfer entropy matrix.
        """
        
        n = self._n
        te = np.zeros([n, n])
        
        data = self._data_mat.copy()
        for i in range(n):
            for j in range(n):
                te[i, j] = self._transfer_entropy_function(
                    data, i, j, bins, shuffle=shuffle)
        
        if save:
            self._te = te.copy()  # store te matrix.
            self._te_min = np.min(te)
            self._te_max = np.max(te)
            self.te = pd.DataFrame(te, columns=self.assets, index=self.assets)
        else:
            return te
        
    def compute_effective_transfer_entropy(self, bins=6, sims=25):
        """
        Compute effective transfer entropy matrix. Returned matrix is
        directional such that asset on X-axis inluences next day transfer
        entropy of asset on the Y-axis.
        
        Parameters
        ----------
        bins: int, default=6
            Number of bins to place log returns in.
        sims: int, default=25
            Number of simulations to use when estimating randomized
            transfer entropy.
        
        Returns
        -------
        ete: [n x n] nd.array
            Effective transfer entropy matrix.
        """
        
        # Compute and store transfer entropy.
        self.compute_transfer_entorpy(bins)
        
        # Compute and store randomized transfer entropy.
        rte_tensor = np.zeros([self._n, self._n, sims])
        for i in range(sims):
            rte_tensor[:, :, i] = self.compute_transfer_entorpy(
                bins, shuffle=True, save=False)
        rte = np.mean(rte_tensor, axis=2)
        self.rte = pd.DataFrame(rte, columns=self.assets, index=self.assets)
        
        # Compute effective transfer entropy.
        self.ete = self.te - self.rte
        
        # Store max and min values.
        self._te_max = np.max(self.te.values)
        self._te_min = np.min(self.ete.values)
        
        
        
    def plot_te(self, te='te', labels=True, cbar=True, ax=None,
                figsize=(6, 6), fontsize=6):
        """
        Plot correlation of assets.
        
        Parameters
        ----------
        te: {'te', 'rte', 'ete'}, default='te'
            Transfer entropy to be plotted.
                - te: transfer entropy
                - rte: randomized transfer entropy
                - ete: effective transfer entropy
        labels: bool, default=True
            If True include labels in plot, else ignore labels.
        cbar: bool, default=True
            If True plot colorbar with scale.
        ax: matplotlib axis, default=None
            Matplotlib axis to plot figure, if None one is created.
        figsize: list or tuple, default=(6, 6)
            Figure size.
        fontsize: int, default=6
            Fontsize for asset labels.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        
        heatmap = eval(f'self.{te}')
        sns.heatmap(heatmap, ax=ax, vmin=self._te_min, vmax=self._te_max,
                    cmap='viridis', xticklabels=labels, yticklabels=labels,
                    cbar=cbar)
        if labels:
            plt.setp(ax.get_xticklabels(), fontsize=fontsize)
            plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    
    
self = TransferEntropy()

# %%
# Set Period.
start = '1/1/2015'
end = '12/31/2016'
self.set_timeperiod(start, end)

# %%
# Find network graph coordinates.
self.build_asset_graph('SLSQP')

#%%
# Plot correlation matrix.
self.plot_corr('spearman')
plt.show()

# %%
# Plot asset graphs for multiple thresholds.
thresholds = [0.5, 0.6, 0.7]

x, y = self._coordinates[:, 0], self._coordinates[:, 1]
adjx = (max(x) - min(x)) / 10  # Give border around graph
adjy = (max(y) - min(y)) / 10  # Give border around graph
n = len(thresholds)
fig, axes = plt.subplots(1, n, figsize=[n*4, 6])

for t, ax in zip(thresholds, axes.flat):
    self.plot_asset_graph(t, thresholds, ax=ax, fontsize=6)
    ax.set_title(f'T = {t}')
    ax.set_xlim(min(x)-adjx, max(x)+adjx)
    ax.set_ylim(min(y)-adjy, max(y)+adjy)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
plt.tight_layout()
plt.show()

# %%
# Plot Transfer Entropy.
self.compute_transfer_entorpy(bins=6)
self.plot_te()
plt.show()

# %%
# Compute Effective Transfer Entropy.
# *** Takes ~10 sec per simulaion with 6 bins ***
self.compute_effective_transfer_entropy(bins=6, sims=5)

# %%
# Plot
fig, axes = plt.subplots(1, 3, figsize=[14, 4])
for te, ax in zip(['te', 'rte', 'ete'], axes.flat):
    self.plot_te(te, labels=False, ax=ax, cbar=False)
    ax.set_title(f'${te.upper()}_{{X \\rightarrow Y}}$')
plt.tight_layout()
plt.show()


# %%
