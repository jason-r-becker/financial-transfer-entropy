from collections import defaultdict
import empiricalutilities as eu
import numpy as np
import pandas as pd

# %%

dates = pd.date_range('1/1/2018', '1/10/2018', freq='B')
X = np.random.randn(1000)/50
Y = np.random.randn(1000)/50
Xt = X[1:]
Xt_1 = X[:-1]
Yt_1 = Y[:-1]
n = len(dates)
df = pd.DataFrame({
    'Date': dates,
    'X': Xt[:n],
    'Xlag': Xt_1[:n],
    'Ylag': Yt_1[:n],
    })
df['Date'] = [f'{x:%m/%d/%y}' for x in dates]
eu.latex_print(df, hide_index=True, prec=4)

b = np.reshape(
        pd.cut(
            np.concatenate([Xt, Xt_1, Yt_1]),
                bins=6,
                labels=False
                ), (3, -1)).T


df1 = pd.DataFrame({
    'Date': dates,
    'X': b[:n,0],
    'Xlag': b[:n,1],
    'Ylag': b[:n,2],
    })
df1['Date'] = [f'{x:%m/%d/%y}' for x in dates]
eu.latex_print(df1, hide_index=True, prec=4)


edges, bins = pd.cut(
                np.concatenate([Xt, Xt_1, Yt_1]),
                bins=6,
                retbins=True
                )

eu.latex_print(bins)

np.concatenate([X, Y]).min()
np.concatenate([X, Y]).max()

for i, b in enumerate(bins):
    d = bins[i+1] - bins[i]
    print(d)
    
