
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import indicators as idc
from util import get_data, plot_data




def build_df_prices(symbols, sd, ed):
    dates = pd.date_range(sd, ed)
    df_prices = get_data(symbols, dates)[symbols]
    
    return pd.DataFrame(df_prices, index=df_prices.index)

def get_rsi(prices, n_days = 20):
    daily_returns = prices[1:] / prices.values[0:-1] - 1 
    dUp, dDown = daily_returns.copy(), daily_returns.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(n_days).mean()
    RolDown = dDown.rolling(n_days).mean().abs()

    RS = RolUp / RolDown
    rsi = 100.0 - (100.0 / (1.0 + RS))
    rsi.columns = ['rsi']
    return rsi

def get_rolling_mean(prices, n_days = 30):
    rmean = pd.Series(data = np.full(len(prices), fill_value= np.NaN), index = prices.index, name = 'rmean')
    for t in range(n_days, len(prices)):
        rmean.iloc[t] = prices[t-n_days:t].values.mean()
    return rmean

def get_rolling_stdev(prices, n_days = 30):
    stdev = pd.Series(data = np.full(len(prices), fill_value= np.NaN), index = prices.index, name = 'stdev')
    for t in range(n_days, len(prices)):
        stdev.iloc[t] = prices[t-n_days:t].values.std()
    return stdev
        
def get_lower_upper_band(rm, rstd):
    upper_band = rm + 2*rstd
    lower_band = rm - 2*rstd
    return lower_band, upper_band 

def get_sma(prices, rmean, n_days = 30, symbol = "JPM"):
    sma  = pd.Series(data = np.full(len(prices), fill_value=np.NaN), index = prices.index, name = 'sma')
    for idx, val in sma[n_days:].iteritems():
        sma.loc[idx] = prices.loc[idx, symbol] / rmean.loc[idx] - 1
    return sma

def get_momentum(prices, n_moment, symbol = "JPM"):
    momentum = pd.Series(data = np.full(len(prices), fill_value=np.NaN), index = prices.index, name ='momentum')
    for i in range(n_moment, len(prices)):
        momentum.iloc[i] = prices.iloc[i][symbol] / prices.iloc[i-n_moment][symbol] - 1
    return momentum

def get_bbands(prices, n_days, symbol = "JPM"):
    rmean = get_rolling_mean(prices, n_days)
    rstd = get_rolling_stdev(prices, n_days)
    bb = pd.Series(data = np.full(len(prices), fill_value= np.NaN), index = prices.index, name='bband')
    for idx, _ in bb[n_days:].iteritems():
        bb.loc[idx] = (prices.loc[idx, symbol] - rmean.loc[idx]) / (2 * rstd.loc[idx]) 
    return bb


def plot_compare_portfolios(df_portfolios, benchmark):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    benchmark.plot(ax = ax, c = 'blue', label = 'benchmark')
    df_portfolios.plot(ax = ax)
    ax.set_xlabel('Dates')
    ax.set_ylabel('Portfolio values')
    ax.grid(True)
    ax.legend(loc = 'best')
    return plt

def plot_strategy(df_prices, df_portval, benchmark, indicator_value, indicator_name, df_buy_signals, df_sell_signals, n_days = 20):
    idx = df_prices.index

    fig = plt.figure(figsize=(11,13))
    ax = fig.add_subplot(2,1,1, title = 'Strategy ' + indicator_name )
    df_prices.plot(ax = ax, color = 'orange')
    ax.set_xticklabels([])
    ax.set_ylabel('Prices')
    ax.vlines(df_buy_signals.index, 0, df_prices.max(), colors='green', label = 'buy signals', linestyles = 'dotted')
    ax.vlines(df_sell_signals.index, 0, df_prices.max(), colors='red', label = 'sell signals', linestyles = 'dotted')
    ax.grid(True)
    ax.legend(loc= 'best')
    
    if indicator_value:
        indicator_value_rmean = get_rolling_mean(indicator_value, n_days)
        bx = fig.add_subplot(3,1,2)
        indicator_value.plot(ax = bx, c = 'purple')
        indicator_value_rmean.plot(ax = bx, c = 'gray')
        bx.set_ylabel(indicator_name)
        bx.set_xticklabels([])
    
        #bx.hlines(1, idx[0], idx[-1], colors='red', label='sell signals')
        #bx.hlines(-1, idx[0], idx[-1], colors='green', label = 'buy signals')
        bx.vlines(df_buy_signals.index, indicator_value.min(), indicator_value.max(), colors='green', label = 'buy signals', linestyles = 'dotted')
        bx.vlines(df_sell_signals.index, indicator_value.min(), indicator_value.max(), colors='red', label = 'sell signals', linestyles = 'dotted')

        bx.grid(True)
        bx.legend(loc= 'best')

    cx = fig.add_subplot(2,1,2)
    benchmark.plot(ax = cx, c = 'blue', label = 'benchmark')
    df_portval.plot(ax = cx, c = 'black')
    cx.set_xlabel('Dates')
    cx.set_ylabel('Portfolio values')
    cx.grid(True)
    cx.legend(loc = 'best')

    return plt



#author adurocher3 Alexis DUROCHER

