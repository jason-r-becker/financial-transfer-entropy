from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize

plt.style.use('fivethirtyeight')
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
    set_timeperiod(start, end): set starting and ending dates for analysis.
    """
    
    def __init__(self, data=None):
        self._all_data = data if data is not None else self._load_data()
        self.data = self._all_data.copy()
        self.assets = list(self.data.columns)
        self._n = len(self.assets)
        
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
        df.dropna(axis=0, inplace=True)  # drop rows with missing data
        df = np.log(df[1:] / df[:-1].values)  # take log returns of prices
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
        data = self._all_data.copy()
        if start:
            data = data[data.index >= pd.to_datetime(start)].copy()
        if start:
            data = data[data.index <= pd.to_datetime(end)].copy()
        self.data = data.copy()
    
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
        
    def build_asset_graph(self):
        """
        Build asset graph of transfer entropy with specified threshold.
        
        Parameters
        ----------

        """
        # Find correlations and distance metric.
        self.corr = self.data.corr('spearman')
        corr = self.corr.values
        self._distance = np.sqrt(2 * (1-corr))
        
        # Solve 2-D coordinate positions.
        opt = minimize(
            self._stress_function,
            x0=np.random.rand(2*self._n),
            # method='SLSQP',
            method='SLSQP',
            tol=1e-6,
            options={'disp': False, 'maxiter': 10000},
            )
        if opt.status != 0:
            raise RuntimeError(opt.message)
        
        self.coordinates = np.reshape(opt.x, (-1, 2))


                
    def plot_asset_graph(self, threshold, ax=None, figsize=(6, 6)):
        """
        Plot asset graph network.
        
        Parameters
        ----------
            threshold: float
                Maximum threshold distance for edges in network.
        ax: matplotlib axis, default=None
            Matplotlib axis to plot figure, if None one is created.
        figsize: list or tuple, default=(6, 6)
            Figure size.
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
        self.nodes = list(set(nodes))
        self.edges = {key: val for key, val in edges.items() if len(val) > 0}
        
        # Store values of edges.
        self.edge_vals = {}
        for node0, node0_edges in edges.items():
            for node1 in node0_edges:
                self.edge_vals[(node0, node1)] = self._distance[node0, node1]
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            
        x = self.coordinates[:, 0]
        y = self.coordinates[:, 1]
        
        # Get sequential colors for edges based on distance.
        cmap = sns.color_palette('magma', 101).as_hex()
        # emin, emax = max(self.edge_vals.values()), min(self.edge_vals.values())
        emin, emax = 0.5, 0.9
        edge_color = {key: cmap[int(100 * (val-emin) / (emax-emin))]
                      for key, val in self.edge_vals.items()}

        # Plot edges.
        for node0, node0_edges in self.edges.items():
            for node1 in node0_edges:
                nodes = [node0, node1]
                ax.plot(x[nodes], y[nodes], c=edge_color[tuple(nodes)],
                        lw=2, alpha=0.7)
        
        # Plot asset names over edges.
        box = {'fill': 'white', 'facecolor': 'white', 'edgecolor': 'k'}
        for node in self.nodes:
            ax.text(x[node], y[node], self.assets[node], bbox=box)
    
# self = TransferEntropy()
# data = self.data
self = TransferEntropy(data)

# %%
# self.set_timeperiod('1/1/2010', '12/31/2010')
self.build_asset_graph()
thresholds = [0.6, 0.7, 0.8]

x, y = self.coordinates[:, 0], self.coordinates[:, 1]
n = len(thresholds)
fig, axes = plt.subplots(1, n, figsize=[n*4, 6])

for t, ax in zip(thresholds, axes.flat):
    self.plot_asset_graph(t, ax)
    ax.set_title(f'T = {t}')
    adj = 0.000001
    ax.set_xlim(min(x)-adj, max(x)+adj)
    ax.set_ylim(min(y)-adj, max(y)+adj)
plt.tight_layout()
plt.show()


# %%
