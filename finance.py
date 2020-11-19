#!/usr/bin/env python

"""fin.py: some financial functions"""

__author__ = "Hua Zhao"

import numpy as np
from scipy.stats import norm
import pandas as pd
from warnings import warn


def iv_bsm_c(s, t, r, k, c, precise=0.001): # 
    """
    Black-Scholes volatility
    """
    v_range = np.arange(0, s, precise)
    pre_diff = float('inf')
    local_mins = dict()
    for v_i in v_range:
        cur_diff = abs(c - bsm(tp='call', s=s, t=t, v=v_i, r=r, k=k))
        if cur_diff <= pre_diff:
            pre_diff = cur_diff
        else:
            local_mins[v_i - precise] = pre_diff
    return min(local_mins, key=local_mins.get)


def bsm(tp, s, t, v, r, k): # 
    """
    Black-Scholes Pricing
    """
    d1 = np.log(s / k) + (r + (v**2 / 2)) * t
    d2 = d1 - v * np.sqrt(t)
    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    n_d1 = norm.cdf(-d1)
    n_d2 = norm.cdf(-d2)
    if tp == 'call':
        return s * nd1 - k * np.exp(-1 * r * t) * nd2
    elif tp == 'put':
        return k * np.exp(-1 * r * t) * n_d2 - s * n_d1
    else:
        raise ValueError(f"no {tp} option type")


def pca(df, thres=0.95, factors=None):
    """
    Principal Component Analysis
    params:
    - df: MxN where M are rows for observations, or in time series analysis, dates; 
          N are columns for features, or in 'forward curve' case, forward months
    """
    if df.shape[0] <= df.shape[1]:
        warn(f"are you sure the matrix you input is MxN where M are examples and N are features?\ncurrent input size: {str(df.shape[0])} * {str(df.shape[1])}")
    C = df.values.T.dot(df.values) / df.shape[0]
    lam, U = np.linalg.eigh(C)
    lam = lam[::-1]  # Reverse so most significant come first.
    U = U[:, ::-1]
    vars_explained = lam.cumsum() / lam.cumsum()[-1]
    if not factors:
        for i, v in enumerate(vars_explained):
            if v >= thres:
                break
        k = i + 1
    else:
        k = factors
    var_explained = vars_explained[k - 1]
    sigmaFull = U * np.sqrt(lam)
    sigma = sigmaFull[:, :k] / np.sqrt(var_explained)
    return pd.DataFrame(sigma, index=df.columns), round(var_explained, 3)


def desea(rtn):
    """deseasonalize financial returns (pandas.dataframe)"""
    lr = rtn.stack()
    df = pd.DataFrame(lr, columns=['log return'])
    df['trade month'] = pd.Index([x.month for x in df.index.get_level_values(0)])
    seas = df.groupby('trade month')['log return'].std()
    seas = seas / seas.mean()
    # seas dones't have 12 months due to lack of data
    if len(seas) < 12:
        res = pd.Series()
        for i in range(1, 13):
            res.loc[i] = seas.get(i, 1)
        rtn = (rtn.T / seas[rtn.index.month].values).T
        return rtn, res
    else:
        rtn = (rtn.T / seas[rtn.index.month].values).T
        return rtn, seas


def test():
    from datain import file
    import os
    rtn = file.hist_fwd(priceIndex='HH', to_rtn=True, roll=True, roll_tenor=22, roll_dropna=True, trailing_year=5, useold=True)
    # rtn, seas = desea(rtn)
    print(rtn)
    lr = rtn.stack()
    df = pd.DataFrame(lr, columns=['log return'])
    df['trade month'] = pd.Index([x.month for x in df.index.get_level_values(0)])
    print(df)


if __name__ == "__main__":
    test()
