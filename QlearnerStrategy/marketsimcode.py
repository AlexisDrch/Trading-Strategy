
import random
import util as ut
import numpy as np
import pandas as pd
import datetime as dt
import indicators as idc
import QLearner as ql
import matplotlib.pyplot as plt
from util import get_data, plot_data



def compute_portfolio_stats(port_val, rfr = 0.0, sf = 252.0, name = 'stats'):
    # daily returns on portfolio value
    daily_rets = (port_val[1:] / port_val[0:-1]) - 1
    daily_rets = daily_rets[1:]
    # compute stats
    cr = (port_val[-1] / port_val[0]) - 1
    adr = daily_rets.mean()
    sddr = daily_rets.std()
    sr = np.sqrt(sf) * ((adr - rfr) / sddr)
    
    return pd.Series(data= [cr[0], adr, sddr, sr], index=['cr', 'adr', 'sddr', 'sr'], name=name)


def normalize_dataframe(dataframe):
    return dataframe/dataframe.iloc[0]

def get_benchmark(df_prices, sv = 100000, impact=0.005, commision=9.95):
    df_trades = pd.DataFrame(data = np.zeros((len(df_prices),2)), 
                             index = df_prices.index, columns=['benchmark', 'cash'])
    # BUY 1000 and hold
    cash_0 = 100000 - 1000 * (df_prices.iloc[0] + df_prices.iloc[0] * impact) - commision
    df_trades.iloc[0] = [1000, cash_0]
    df_holdings = df_trades.cumsum(axis = 0)
    df_holdings.benchmark *= df_prices
    df_portval = df_holdings.sum(axis = 1)
    df_portval = pd.DataFrame(data = df_portval, index = df_prices.index, columns=['benchmark'])
    return df_portval


def compute_portvals(df_trades, name = 'portfolio',\
    start_val = 1000000, commission=9.95, impact=0.005):
    
    symbol = df_trades.columns[0]
    # getting corresponding porfolio
    sd = df_trades.index[0].to_datetime()
    ed = df_trades.index[-1].to_datetime()

    df_prices_all = get_data([symbol], pd.date_range(sd, ed))
    df_prices = df_prices_all[symbol]
    
    df_trades.columns = ['trades']
    portfolio_df = pd.concat([df_prices, df_trades], axis = 1)
    portfolio_df['impact'] = impact
    commissions = [commission if trade != 0 else 0 for trade in df_trades.values]
    impacts = [impact if trade > 0 else -impact if trade < 0 else 0 for trade in df_trades.values]
    portfolio_df['impacted_price'] = df_prices + df_prices *  impacts 
    portfolio_df['cash'] = - portfolio_df.impacted_price * portfolio_df.trades
    portfolio_df['cash'] = portfolio_df.cash.cumsum(axis = 0) + start_val - commissions
    portfolio_df['cum_trades'] = df_trades.cumsum(axis = 0)
    portfolio_df['holding'] = portfolio_df.cum_trades * portfolio_df.impacted_price
    
    port_val = pd.DataFrame(portfolio_df.holding + portfolio_df.cash, columns=[symbol])
    portfolio_df = pd.concat([portfolio_df, port_val], axis = 1)
    portfolio_df['daily_ret'] = port_val[1:] / port_val.values[0:-1] - 1
    return port_val

#author adurocher3 Alexis DUROCHER