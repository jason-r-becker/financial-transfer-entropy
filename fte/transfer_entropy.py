import json
import warnings
from collections import Counter, defaultdict
from glob import glob

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm
plt.style.use('fivethirtyeight')



# %matplotlib qt
# %%

class TransferEntropy:
    """
    Class to compute asset graphs using transfer entropy.

    Parameters
    ----------
    assets: list[str], default=None
        List of assets to use from loaded data.
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
    subset_assets(assets): Subset assets used.
    build_asset_graph(solver): Find graph coordinates.
    compute_transfer_entorpy(bins): Return transfer entropy.
    compute_effective_transfer_entropy: Return effective transfer entropy.
    plot_asset_graph(threshold): Plot asset graph edges that meet threshold.
    plot_corr(method): Plot correlations between assets.
    plot_te(te): Plot transfer entropy heatmap.
    """

    def __init__(self, assets=None, data=None):
        self.assets = assets
        self._prices = data if data is not None else self._load_data()
        self.set_timeperiod('1/3/2011', '12/31/2018')
        
    def _read_file(self, fid):
        """Read data file as DataFrame."""
        df = pd.read_csv(
            fid,
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
            )
        return df

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
        sorted(list(data))
        # Ignore warnings for missing data.
        warnings.filterwarnings('ignore')

        # Subset data by time period.
        if start is not None:
            data = data[data.index >= pd.to_datetime(start)].copy()
        if end is not None:
            data = data[data.index <= pd.to_datetime(end)].copy()
        
        # Drop Weekends and forward fill Holidays.
        keep_ix = [ix.weekday() < 5 for ix in list(data.index)]
        data = data[keep_ix].copy()
        data.fillna(method='ffill', inplace=True)
        
        # Calculate Log Returns.
        self.data = np.log(data[1:] / data[:-1].values)  # log returns
        self.data.dropna(axis=1, inplace=True)  # Drop assets with missing data.
        
        # Map asset names to DataFrame.
        # with open('../data/asset_mapping.json', 'r') as fid:
        #     asset_map = json.load(fid)
        # self.assets = [asset_map.get(a, a) for a in list(self.data)]
        # self._n = len(self.assets)

        # Subset data to specified assets.
        if self.assets is not None:
            self.subset_assets(self.assets)
        else:
            self.assets = list(self.data)
        self._n = len(self.assets)
        # Rename DataFrame with asset names and init data matrix.
        # self.data.columns = self.assets
        self._data_mat = self.data.values

    def subset_assets(self, assets):
        """
        Subset data to specified assets.
        
        Parameters
        ----------
        assets: list[str]
            List of assets to use.
        """
        self.data = self.data[assets].copy()
        self.assets = assets
        self._n = len(self.assets)
        
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

    def plot_corr(self, method='pearson', ax=None, figsize=(6, 6),
                  fontsize=8, cbar=True, labels=True):
        """
        Plot correlation of assets.

        Parameters
        ----------
        method: {'spearman', 'pearson', 'kendall'}, default='pearson'
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
        cbar: bool, default=True
            If True include color bar.
        labels: bool, default=False
            If True include tick labels for x & y axis.
            If False do not inlclude labels.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        self._corr = self.data.corr(method)
        sns.heatmap(self._corr, ax=ax, vmin=-1, vmax=1, cbar=cbar,
                    xticklabels=labels, yticklabels=labels, cmap='coolwarm')
        if labels:
            plt.setp(ax.get_xticklabels(), fontsize=fontsize)
            plt.setp(ax.get_yticklabels(), fontsize=fontsize)
            
        if cbar:
            cbar_ax = ax.collections[0].colorbar
            ticks = np.linspace(-1, 1, 5)
            cbar_ax.set_ticks(ticks)
            cbar_ax.set_ticklabels(ticks)

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
        data_matrix = np.concatenate([x, x_lag, y_lag])
        bin_edges = np.concatenate([
            np.linspace(np.min(data_matrix), 0, int(bins/2)+1),
            np.linspace(0, np.max(data_matrix), int(bins/2)+1)[1:],
            ])
        bin_vals = np.reshape(pd.cut(
            data_matrix, bins=bin_edges, labels=False), (3, -1)).T

        if shuffle:
            # Shuffle y_lag for randomized transfer entropy.
            np.random.shuffle(bin_vals[:, 2])

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

    def compute_transfer_entropy(self, bins=6, shuffle=False, save=True):
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

    def compute_effective_transfer_entropy(self, bins=6, sims=25,
                                           std_threshold=1, pbar=False):
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
        pbar: bool, default=False
            If True, show progress bar for simulations.
            
        Returns
        -------
        ete: [n x n] nd.array
            Effective transfer entropy matrix.
        """

        # Compute and store transfer entropy.
        self.compute_transfer_entropy(bins)

        # Compute and store randomized transfer entropy.
        self.rte_tensor = np.zeros([self._n, self._n, sims])
        if pbar:
            for i in tqdm(range(sims)):
                self.rte_tensor[:, :, i] = self.compute_transfer_entropy(
                    bins, shuffle=True, save=False)
        else:
            for i in range(sims):
                self.rte_tensor[:, :, i] = self.compute_transfer_entropy(
                    bins, shuffle=True, save=False)
        
        # Peform significance test on ETE values.
        cutoff = norm(0, 1).cdf(std_threshold)
        ete = np.zeros([self._n, self._n])
        for i in range(self._n):
            for j in range(self._n):
                if i == j:
                    continue
                te = self.te.iloc[i, j]
                rte_array = self.rte_tensor[i, j, :]
                z = norm(np.mean(rte_array), np.std(rte_array)).cdf(te)
                if z > cutoff:
                    ete[i, j] = te - np.mean(rte_array)

        rte = np.mean(self.rte_tensor, axis=2)
        self.rte = pd.DataFrame(rte, columns=self.assets, index=self.assets)
        self.ete = pd.DataFrame(ete, columns=self.assets, index=self.assets)
        
        # Store max and min values.
        self._te_max = np.max(self.te.values)
        self._te_min = np.min(self.ete.values)

    def plot_te(self, te='ete', labels=True, cbar=True, ax=None,
                figsize=(6, 6), fontsize=6, vmin=None, vmax=None):
        """
        Plot correlation of assets.

        Parameters
        ----------
        te: {'te', 'rte', 'ete'}, default='ete'
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

        vmin = self._te_min if vmin is None else vmin
        vmax = self._te_max if vmax is None else vmax
        
        heatmap = eval(f'self.{te}')
        sns.heatmap(heatmap, ax=ax, vmin=vmin, vmax=vmax,
                    cmap='viridis', xticklabels=labels, yticklabels=labels,
                    cbar=cbar)
        if labels:
            plt.setp(ax.get_xticklabels(), fontsize=fontsize)
            plt.setp(ax.get_yticklabels(), fontsize=fontsize)

    def plot_corr_network(self, nx_type='circle', threshold=0.8, pos=None,
        method='pearson', ax=None, figsize=(12, 12), cmap='Reds', fontsize=8):
        """
        Plot correlation of assets.

        Parameters
        ----------
        nx_type: {'circle', 'cluster'}, default='circle'
            Type of network graph.
                - 'circle': Circular graph in decreasing node strength order.
                - 'cluster': Spring graph of clusters.
        threshold: int, default=0.9
            Lower threshold of link strength for connections in network graph.
        pos: dict, default=None
            Dictionary with nodes as keys and positions as values.
        method: {'spearman', 'pearson', 'kendall'}, default='spearman'
            Correlation method.
                - 'spearman': Spearman rank correlation
                - 'pearson': standard correlation coefficient
                - 'kendall': Kendall Tau correlation coefficient
        ax: matplotlib axis, default=None
            Matplotlib axis to plot figure, if None one is created.
        figsize: list or tuple, default=(6, 6)
            Figure size.
        cmap: str, default='Blues'
            Matploblib cmap.
        fontsize: int, default=8
            Fontsize for asset labels.
        """
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Find correlation matrix and transform it in a links data frame.
        corr = self.data.corr(method).abs()
        links = corr.stack().reset_index()
        links.columns = ['in', 'out', 'weight']
        links = links.loc[(links['in'] != links['out'])
                          & (links['weight'] > threshold)]
        links = links[['out', 'in', 'weight']].reset_index(drop=True)

        # Subset to only use upper triangle portion of correlation matrix.
        edges = defaultdict(int)
        ix = []
        for i, row in links.iterrows():
            if (edges[(row['out'], row['in'])]
                + edges[(row['in'], row['out'])] > 0):
                continue
            else:
                edges[(row['out'], row['in'])] += 1
                ix.append(i)
        links = links.loc[ix]

        # Sort by node centrality, and minmax range.
        nc = np.sum(corr, axis=1) - 1
        nc = (nc-np.min(nc)) / (np.max(nc)-np.min(nc))
        nc.sort_values(inplace=True, ascending=False)
        nx_nodes = list(set(list(links['out']) + list(links['in'])))
        node_ix = [node in nx_nodes for node in nc.index]
        nc = nc.loc[node_ix]
        node_strengths = nc.values
        nodes = list(nc.index)
    
        # Build OrderedGraph of nodes by centrality measure.
        G = nx.OrderedGraph()
        G.add_nodes_from(nodes)
        w = links['weight'].values
        lwidths = np.round(0.3 + 3*((w - np.min(w)) / (np.max(w)-np.min(w))), 2)
        edge_tuples = list(links[['in', 'out']].itertuples(index=False))
        for edge, w, lw in zip(edge_tuples, links['weight'].values, lwidths):
            G.add_edge(*edge, weight=w, lw=lw)
            
        # Get graph position.
        if pos is None:
            pos = {
                'circle': nx.circular_layout,
                'cluster': nx.spring_layout,
                }[nx_type](G)
        self.pos = pos
        
        # Draw network edges.
        edge_kwargs = {
            'ax': ax,
            'pos': pos,
            'edge_cmap': plt.get_cmap(cmap),
            'alpha': 0.7,
            'edge_vmin': 0.8 * np.min(links['weight']),
            'edge_vmax': np.max(links['weight']),
            }
        nx_edges = sorted(G.edges(data=True), key=lambda x: x[-1]['weight'])
        for u, v, d in nx_edges:
            edge_kwargs['edge_color'] = np.array([d['weight']])
            edge_kwargs['width'] = np.array([d['lw']])
            nx.draw_networkx_edges(G, edgelist=[(u,v)], **edge_kwargs)
            
        # Draw network nodes and labels.
        vmax = 1.1 * np.max(nc)
        vmin = 0.9 * nc.iloc[int(0.7*len(nc))]
        node_colors = [max(ns, vmin*1.05) for ns in node_strengths]
        nx.draw_networkx_nodes(
            G, ax=ax, pos=pos, cmap=cmap, node_size=1000*node_strengths,
            node_color=node_colors, alpha=0.9, vmax=vmax, vmin=vmin)
        nx.draw_networkx_labels(
            G, ax=ax, pos=pos, font_weight='bold', font_size=fontsize)
        
        # Hide axis ticks and grid.
        ax.grid(False)
        ax.tick_params(
            bottom=False, left=False, labelbottom=False, labelleft=False)

    def plot_ete_network(self, nx_type='circle', node_value='out', pos=None,
        threshold=0.01,  ax=None, figsize=(12, 12), cmap='Blues', fontsize=8):
        """
        Plot correlation of assets.

        Parameters
        ----------
        nx_type: {'circle', 'cluster'}, default='circle'
            Type of network graph.
                - 'circle': Circular graph in decreasing node strength order.
                - 'cluster': Spring graph of clusters.
        node_value: {'out', 'in'}, default='out'
            Transfer entropy node centraily measure.
                'out': Total transfer entropy out of each node.
                'in': Total transfer entropy in to each node.
        pos: dict, default=None
            Dictionary with nodes as keys and positions as values.
        threshold: int, default=0.01
            Lower threshold of link strength for connections in network graph.
        ax: matplotlib axis, default=None
            Matplotlib axis to plot figure, if None one is created.
        figsize: list or tuple, default=(6, 6)
            Figure size.
        cmap: str, default='Blues'
            Matploblib cmap.
        fontsize: int, default=8
            Fontsize for asset labels.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Compute effective transfer entropy DataFrame and
        # transform it in a links data frame.
        links = self.ete.stack().reset_index()
        links.columns = ['in', 'out', 'weight']
        links = links.loc[links['weight'] > threshold]
        links = links[['out', 'in', 'weight']]
        if len(links) < 1:
            msg = 'Threshold is too high, lower threshold below '
            msg += f'{np.max(self.ete.values):.4f}'
            raise ValueError(msg)
            
        # Sort by node centrality, and minmax range.
        axis = {'in': 1, 'out': 0}[node_value]
        nc = np.sum(self.ete, axis=axis)
        nc = (nc-np.min(nc)) / (np.max(nc)-np.min(nc))
        nc.sort_values(inplace=True, ascending=False)
        nx_nodes = list(set(list(links['out']) + list(links['in'])))
        node_ix = [node in nx_nodes for node in nc.index]
        nc = nc.loc[node_ix]
        node_strengths = 0.1 + 0.9*nc.values
        nodes = list(nc.index)
        
        # Build OrderedGraph of nodes by centrality measure.
        G = nx.OrderedDiGraph()
        G.add_nodes_from(nodes)
        w = links['weight'].values
        lwidths = np.round(0.1 + 3*((w - np.min(w)) / (np.max(w)-np.min(w))), 2)
        edge_tuples = list(links[['in', 'out']].itertuples(index=False))
        for edge, w, lw in zip(edge_tuples, links['weight'].values, lwidths):
            G.add_edge(*edge, weight=w, lw=lw)

        # Get graph position.
        if pos is None:
            pos = {
                'circle': nx.circular_layout,
                'cluster': nx.spring_layout,
                }[nx_type](G)
        self.pos = pos

        # Draw network edges.
        edge_kwargs = {
            'ax': ax,
            'pos': pos,
            'arrowstyle': '-|>',
            'arrowsize': 15,
            'edge_cmap': plt.get_cmap(cmap),
            'alpha': 0.6,
            'edge_vmin': 0.8 * np.min(links['weight']),
            'edge_vmax': np.max(links['weight']),
            }

        nx_edges = sorted(G.edges(data=True), key=lambda x: x[-1]['weight'])
        for u, v, d in nx_edges:
            edge_kwargs['edge_color'] = np.array([d['weight']])
            edge_kwargs['width'] = np.array([d['lw']])
            nx.draw_networkx_edges(G, edgelist=[(u,v)], **edge_kwargs)
            
        # Draw network nodes and labels.
        vmax = 1.2 * np.max(nc)
        # vmin = 0.9 * nc.iloc[int(0.5*len(nc))]
        vmin = -0.1
        
        node_colors = [max(ns, vmin*1.5) for ns in node_strengths]
        nx.draw_networkx_nodes(
            G, ax=ax, pos=pos, cmap=cmap, node_size=1000*node_strengths,
            node_color=node_colors, alpha=0.9, vmax=vmax, vmin=vmin)
        nx.draw_networkx_labels(
            G, ax=ax, pos=pos, font_weight='bold', font_size=fontsize)
        
        # Hide axis ticks and grid.
        ax.grid(False)
        ax.tick_params(
            bottom=False, left=False, labelbottom=False, labelleft=False)

eqs = 'SPY DIA XLK XLV XLF IYZ XLY XLP XLI XLE XLU XME IYR XLB XPH IWM PHO ' \
    'SOXX WOOD FDN GNR IBB ILF ITA IYT KIE PBW ' \
    'AFK EZA ECH EWW EWC EWZ EEM EIDO EPOL EPP EWA EWD EWG EWH EWJ EWI EWK ' \
    'EWL EWM EWP EWQ EWS EWT EWU EWY GXC HAO EZU RSX TUR'.split()
fi = 'AGG SHY IEI IEF TLT TIP LQD HYG MBB'.split()
cmdtys = 'GLD SLV DBA DBC USO UNG'.split()
assets = eqs + fi + cmdtys


#
# self = TransferEntropy(assets=assets[::2])
#
# # Set Period.
# start = '1/2/2011'
# end = '12/31/2018'
# self.set_timeperiod(start, end)
# #
# # %%
# # Plot correlation matrix.
# self.plot_corr(cbar=True)
# plt.show()
#
# %%
# corr_thresh = 0.75
# fig, axes = plt.subplots(1, 2, figsize=(12, 12), sharex=True, sharey=True)
# fig.suptitle('Correlation Magnitude')
# self.plot_corr_network(
#     nx_type='circle',
#     threshold=corr_thresh,
#     ax=axes[0],
#     )
# self.plot_corr_network(
#     nx_type='cluster',
#     threshold=corr_thresh,
#     ax=axes[1],
#     )
# # %%
# # Find network graph coordinates.
# corr_thresh = 0.5
# self.plot_corr_network(nx_type='circle', threshold=corr_thresh)
# plt.show()
#
# self.plot_corr_network(nx_type='cluster', threshold=corr_thresh)
# plt.show()
#
# plt.plot(nc)
# # %%
# Compute effective transfer entropy.
# self.compute_effective_transfer_entropy(sims=5, pbar=True)

# ete = self.ete
# %%

# # %%
# Plot effective transfer entropy.
# self.plot_te(te='ete', vmax=0.5*np.max(self.te.values))
# plt.show()

# # %%
# self.ete = ete.copy()
#
# # Plot effective transfer entropy out.
# np.max(self.ete.values)
# ete_thresh_high = 0.018
# ete_thresh_low = 0.012
#
# fig, axes = plt.subplots(1, 3, figsize=(20, 10))
# fig.suptitle('Transfer Entropy Out')
# self.plot_ete_network(
#     nx_type='circle',
#     node_value='out',
#     threshold=ete_thresh_low,
#     ax=axes[0],
#     )
# self.plot_ete_network(
#     nx_type='cluster',
#     node_value='out',
#     threshold=ete_thresh_low,
#     ax=axes[1],
#     )
# axes[1].set_title('Low Threshold')
# self.plot_ete_network(
#     nx_type='cluster',
#     node_value='out',
#     threshold=ete_thresh_high,
#     # pos=self.pos,
#     ax=axes[2],
#     )
# axes[2].set_title('High Threshold')
# plt.show()
#
# # Plot effective transfer entropy in.
# fig, axes = plt.subplots(1, 3, figsize=(20, 10))
# fig.suptitle('Transfer Entropy In')
# self.plot_ete_network(
#     nx_type='circle',
#     node_value='in',
#     threshold=ete_thresh_low,
#     cmap='Purples',
#     ax=axes[0],
#     )
# self.plot_ete_network(
#     nx_type='cluster',
#     node_value='in',
#     threshold=ete_thresh_low,
#     cmap='Purples',
#     ax=axes[1])
# axes[1].set_title('Low Threshold')
# self.plot_ete_network(
#     nx_type='cluster',
#     node_value='in',
#     threshold=ete_thresh_high,
#     cmap='Purples',
#     # pos=self.pos,
#     ax=axes[2])
# axes[2].set_title('High Threshold')
# plt.show()
# %%
# # Find network graph coordinates.
# thresh = 0.03
# self.plot_ete_network(nx_type='circle', threshold=thresh)
# plt.show()
#
# self.plot_ete_network(nx_type='cluster', threshold=thresh)
# plt.show()
#
# # %%
# self.plot_ete_network(nx_type='circle', node_value='in', threshold=thresh,
#                       cmap='Purples')
# plt.show()
#
# self.plot_ete_network(nx_type='cluster', node_value='in', threshold=thresh,
#                       cmap='Purples')
# plt.show()

# %%
# --------------------- Still in-progress work below ----------------------- #

# # %%
# # Plot asset graphs for multiple thresholds.
# thresholds = [0.5, 0.6, 0.7]
#
# x, y = self._coordinates[:, 0], self._coordinates[:, 1]
# adjx = (max(x) - min(x)) / 10  # Give border around graph
# adjy = (max(y) - min(y)) / 10  # Give border around graph
# n = len(thresholds)
# fig, axes = plt.subplots(1, n, figsize=[n*4, 6])
#
# for t, ax in zip(thresholds, axes.flat):
#     self.plot_asset_graph(t, thresholds, ax=ax, fontsize=6)
#     ax.set_title(f'T = {t}')
#     ax.set_xlim(min(x)-adjx, max(x)+adjx)
#     ax.set_ylim(min(y)-adjy, max(y)+adjy)
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
#     ax.grid(False)
# plt.tight_layout()
# plt.show()
#
#


# # %%
# # Plot
# fig, axes = plt.subplots(1, 3, figsize=[14, 4])
# for te, ax in zip(['te', 'rte', 'ete'], axes.flat):
#     self.plot_te(te, labels=False, ax=ax, cbar=False)
#     ax.set_title(f'${te.upper()}_{{X \\rightarrow Y}}$')
# plt.tight_layout()
# plt.show()
#
#
