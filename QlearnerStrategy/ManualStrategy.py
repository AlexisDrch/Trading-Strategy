
# coding: utf-8

# In[32]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime as dt
from util import get_data, plot_data
import indicators as idc


# In[71]:

symbol = 'JPM'
sd=dt.datetime(2008,1,1)
ed=dt.datetime(2009, 12, 31)
sd_test=dt.datetime(2010,1,1)
ed_test=dt.datetime(2011, 12, 31)
sv = 100000


# In[102]:

def grid_search(df_prices):
    stats = pd.Series(np.zeros(4), index=['cr', 'adr', 'sddr', 'sr'], name='0')
    range_bbdays = [5, 10, 20]
    range_ndays_momentum = [5, 10, 20]
    range_ndays_std = [5, 10, 20]
    range_ndays_rsi = [5, 10, 20, 30]
    range_tresh_bband = [0.8,0.9,1] #[0.5, 0.6, 0.7, 0.8, 0.9, 1]
    range_tresh_momentum = [0.2, 0.3, 0.4]
    range_tresh_std_low = [0.0,  1, 2,  3, 5]
    range_tresh_std_high = [0.0,  1, 2,  3, 5]
    for bb_days in range_bbdays:
        print('ya')
        for mm_days in range_ndays_momentum:
            for std_days in range_ndays_std:
                for rsi_days in range_ndays_rsi:
                    for bb_tr in range_tresh_bband:
                        for mm_tr in range_tresh_momentum:
                            for std_tr_low in range_tresh_std_low:
                                for std_tr_high in range_tresh_std_high:
                                    name = str(bb_days)+'/'+str(mm_days)+'/'+str(std_days)+'/'+str(rsi_days)+'/'+str(bb_tr)+'/'+str(mm_tr)+'/'+str(std_tr_low)+'/'+str(std_tr_high)
                                    df_trades, bband = testPolicyMixed(
                                        bb_days, mm_days, std_days, rsi_days, bb_tr, mm_tr, std_tr_low, std_tr_high,
                                        symbol='JPM', sd=dt.datetime(2008, 1, 1, 0, 0),
                                        ed=dt.datetime(2009, 12, 31, 0, 0), sv=100000
                                    )
                                    portfolio = normalize_dataframe(
                                        get_portfolio_value(df_prices, df_trades,'JPM', name)
                                    )
                                    stats_porfolio = compute_portfolio_stats(portfolio.values, name=name)
                                    stats = pd.concat([stats, stats_porfolio], axis = 1)

                
    return stats

def compute_portfolio_stats(port_val, rfr = 0.0, sf = 252.0, name = 'stats'):
    # daily returns on portfolio value
    daily_rets = (port_val[1:] / port_val[0:-1]) - 1
    daily_rets = daily_rets[1:]
    # compute stats
    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = np.sqrt(sf) * ((adr - rfr) / sddr)
    
    return pd.Series(data= [cr, adr, sddr, sr], index=['cr', 'adr', 'sddr', 'sr'], name=name)


def normalize_dataframe(dataframe):
    return dataframe/dataframe.iloc[0]

def get_benchmark(df_prices, sv = 100000, impact=0.005, commision=9.95):
    df_trades = pd.DataFrame(data = np.zeros((len(df_prices),2)), 
                             index = df_prices.index, columns=['benchmark', 'cash'])
    # BUY 1000 JPM and hold
    df_trades.iloc[0] = [1000, 100000 - 1000 * (df_prices.iloc[0] + df_prices.iloc[0] * impact) - commision]
    df_holdings = df_trades.cumsum(axis = 0)
    df_holdings.benchmark *= df_prices
    df_portval = df_holdings.sum(axis = 1)
    df_portval = pd.DataFrame(data = df_portval, index = df_prices.index, columns=['benchmark'])
    return df_portval

def get_portfolio_value(df_prices, df_trades, symbol,name = 'rule-based portfolio', impact=0.005, commision=9.95):
    best_df = pd.concat([df_prices, df_trades], axis = 1)
    best_df['impact'] = impact
    commisions = [commision if trade != 0 else 0 for trade in df_trades.values]
    impacts = [impact if trade > 0 else -impact if trade < 0 else 0 for trade in df_trades.values]
    print(len(impacts), len(df_prices))
    best_df['impacted_price'] = df_prices + df_prices *  impacts 
    best_df['cash'] = - best_df.impacted_price * best_df.trades
    best_df['cash'] = best_df['cash'].cumsum(axis = 0) + sv - commisions
    best_df['holding'] = best_df.trades.cumsum(axis = 0) * best_df.impacted_price
    port_val = pd.DataFrame(best_df.holding + best_df.cash, columns=[name])
    return port_val

def crossed_from_above(crossed, crosser, t):
    return (crosser[t-1] > crossed[t]) & (crosser[t] <= crossed[t])

def crossed_from_below(crossed, crosser, t):
    return (crosser[t-1] < crossed[t]) & (crosser[t] >= crossed[t])

def get_portfolios_dataframe(policy, df_prices, range_ndays):
    df_trades, bband = policy(range_ndays[0])
    portfolios = normalize_dataframe(get_portfolio_value(df_prices, df_trades, 'JPM', str(range_ndays[0])))
    for n_days in range_ndays[1:]:
        df_trades, bband = policy(n_days)
        portfolio = normalize_dataframe(get_portfolio_value(df_prices, df_trades, 'JPM', str(n_days)))
        portfolios = pd.concat([portfolios, portfolio], axis = 1)
    
    return portfolios

def get_portfolios_dataframe_tresh(policy, df_prices, range_tresh):
    df_trades, bband = policy(20, range_tresh[0])
    portfolios = normalize_dataframe(get_portfolio_value(df_prices, df_trades, 'JPM', str(range_tresh[0])))
    for n_tresh in range_tresh[1:]:
        df_trades, bband = policy(20, n_tresh)
        portfolio = normalize_dataframe(get_portfolio_value(df_prices, df_trades, 'JPM', str(n_tresh)))
        portfolios = pd.concat([portfolios, portfolio], axis = 1)
    
    return portfolios

def testPolicyBBand(n_days, symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009, 12, 31), sv = 100000):
    df_prices_all = get_data([symbol], pd.date_range(sd, ed))
    df_prices = df_prices_all[symbol]
    df_trades = pd.DataFrame(data = np.zeros(len(df_prices)), index = df_prices.index, columns=['trades'])
    
    bband = idc.get_bbands(pd.DataFrame(df_prices, index = df_prices.index, columns = [symbol]),  n_days)
    tresh = 1
    # use bolinger band @try to vectorize ? 
    for t in range(1, len(df_prices)):
        df_net_holdings = df_trades.cumsum(axis = 0)
        net_holdings = df_net_holdings.values[t]
        if (net_holdings > 1000) | (net_holdings < -1000):
            print("ERROR")
        # buy signal 
        if (bband[t-1] <= - tresh) & (bband[t] > -  tresh):
            if (net_holdings == 0):
                df_trades.iloc[t] = 1000
            elif (net_holdings == -1000):
                df_trades.iloc[t] = 2000
        # sell signal
        if (bband[t-1] >= tresh) & (bband[t] < tresh):
            if (net_holdings == 0):
                df_trades.iloc[t] = -1000
            elif (net_holdings == 1000):
                df_trades.iloc[t] = -2000
    return df_trades, bband


def testPolicyMixed(n_days_bb=20, n_days_mtum=20, n_days_std=20,
                    n_days_rsi = 20,
                    tresh_bb=1, tresh_mmtum = 0.2, tresh_std_low=1, tresh_std_high=1,
                    symbol = "JPM", sd=dt.datetime(2008,1,1),
                    ed=dt.datetime(2009, 12, 31), sv = 100000):
    df_prices_all = get_data([symbol], pd.date_range(sd, ed))
    df_prices = df_prices_all[symbol]
    df_trades = pd.DataFrame(data = np.zeros(len(df_prices)), index = df_prices.index, columns=['trades'])
    
    #rstd = idc.get_rolling_stdev(pd.DataFrame(df_prices, index = df_prices.index, columns = [symbol]),  n_days_std)
    rstd = pd.rolling_std(df_prices, window= n_days_std)
    bband = idc.get_bbands(pd.DataFrame(df_prices, index = df_prices.index, columns = [symbol]),  n_days_bb)
    momentum = idc.get_momentum(pd.DataFrame(df_prices, index = df_prices.index, columns = [symbol]),  n_days_mtum)
    abs_momentum = momentum.abs()
    rsi = idc.get_rsi(df_prices, n_days_rsi)
    # use rsi to generat sell & buy signal:
        # > 70 : overbought : should sell 
        # < 30 : oversold : should buy
    #use volatility as a filter:
        # if std to low, time to buy, std will increase soon, # maybe make a cross over on the rmean of std ?
        # if std to high, means the growth was strong but will back to the mean
    # use momentum to confirm sell & buy signal:
        # confirmed if small enough momentum 
        # (changes in prices is slowing - time to take action)
    for t in range(1, len(df_prices)-1):
        df_net_holdings = df_trades.cumsum(axis = 0)
        net_holdings = df_net_holdings.values[t]
        if (net_holdings > 1000) | (net_holdings < -1000):
            print("ERROR")
        
        # buy signal   
        if ((bband[t-1] <= -tresh_bb) & (bband[t] > -tresh_bb)) | (rsi.iloc[t] < 30) :
            if (abs_momentum[t] <= tresh_mmtum):
                if rstd[t] <= tresh_std_low:
                    if (net_holdings == 0):
                        df_trades.iloc[t] = 1000
                    elif (net_holdings == -1000):
                        df_trades.iloc[t] = 2000
        # sell signal
        if ((bband[t-1] >= tresh_bb) & (bband[t] < tresh_bb))| (rsi.iloc[t] > 70) :
            if (abs_momentum[t] <= tresh_mmtum):
                if rstd[t] >= tresh_std_high: # could be remove since thresh = 0
                    if (net_holdings == 0):
                        df_trades.iloc[t] = -1000
                    elif (net_holdings == 1000):
                        df_trades.iloc[t] = -2000
    return df_trades, rstd


def testPolicyMomentum(n_days_bb=20, n_days_mtum=20, n_days_std=20,
                    tresh_bb=1, tresh_mmtum = 0.2, tresh_std_low=1, tresh_std_high=1,
                    symbol = "JPM", sd=dt.datetime(2008,1,1),
                    ed=dt.datetime(2009, 12, 31), sv = 100000):
    df_prices_all = get_data([symbol], pd.date_range(sd, ed))
    df_prices = df_prices_all[symbol]
    df_trades = pd.DataFrame(data = np.zeros(len(df_prices)), index = df_prices.index, columns=['trades'])
    
    rstd = idc.get_rolling_stdev(pd.DataFrame(df_prices, index = df_prices.index, columns = [symbol]),  n_days_std)
    bband = idc.get_bbands(pd.DataFrame(df_prices, index = df_prices.index, columns = [symbol]),  n_days_bb)
    momentum = idc.get_momentum(pd.DataFrame(df_prices, index = df_prices.index, columns = [symbol]),  n_days_mtum)
    abs_momentum = momentum.abs()
    # use momentum to confirm sell & buy signal:
        # confirmed if small enough momentum 
        # (changes in prices is slowing - time to take action)
    for t in range(1, len(df_prices)):
        df_net_holdings = df_trades.cumsum(axis = 0)
        net_holdings = df_net_holdings.values[t]
        if (net_holdings > 1000) | (net_holdings < -1000):
            print("ERROR")
        #if rstd[t] >= tresh_std:
            # buy signal 
        if (bband[t-1] <= -tresh_bb) & (bband[t] > -tresh_bb):
            if (abs_momentum[t] <= tresh_mmtum):
                if (net_holdings == 0):
                    df_trades.iloc[t] = 1000
                elif (net_holdings == -1000):
                    df_trades.iloc[t] = 2000
        # sell signal
        if (bband[t-1] >= tresh_bb) & (bband[t] < tresh_bb):
            if (abs_momentum[t] <= tresh_mmtum):
                if (net_holdings == 0):
                    df_trades.iloc[t] = -1000
                elif (net_holdings == 1000):
                    df_trades.iloc[t] = -2000
    return df_trades, momentum


#author adurocher3 Alexis DUROCHER

