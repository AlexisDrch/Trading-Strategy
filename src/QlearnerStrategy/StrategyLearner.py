
import random
import util as ut
import numpy as np
import pandas as pd
import datetime as dt
import indicators as idc
import QLearner as ql
import matplotlib.pyplot as plt
from util import get_data, plot_data

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False, impact=0.0, \
        max_epoch = 1000, rar = 0.9, radr = 0.99, alpha = 0.2, gamma = 0.9):
        self.verbose = verbose
        self.impact = impact
        self.window_size = 10
        self.n_actions = 3
        self.signal_values = {0: 0, 1: -1000, 2: 1000}
        self.max_epoch = max_epoch
        self.net_holdings = 0
        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma

    def normalizeIndicators(self, X):
        """
        finally, not used : prefer pd.qcut to split uniformly in each bin. @todo : remove
        """
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * 9
        return pd.Series(np.around(X_scaled), dtype = np.int64)
        
    def buildIndicators(self, df_prices, symbol = "IBM"):
        """
        Build (panda Series) indicators from the prices dataframe
        @params df_prices: the prices dataframe
        @params symbol: the asset's symbol
        5,10,20,30
        """
        # rolling bollinger band
        self.bband = idc.get_bbands(pd.DataFrame(df_prices, index = df_prices.index,
                                                 columns = [symbol]), self.window_size, symbol = symbol).dropna()
        # fixing treshold for consistencies before normalization
        self.bband[self.bband < -1] = -1
        self.bband[self.bband > 1] = 1
        self.discretize_bband = self.normalizeIndicators(self.bband)
        
        # rolling momentum abs
        self.momentum = idc.get_momentum(pd.DataFrame(df_prices, index = df_prices.index,
                                                 columns = [symbol]), self.window_size, symbol = symbol).dropna()
        self.momentum[self.momentum < -0.3] = -0.3
        self.momentum[self.momentum > 0.3] = 0.3
        self.discretize_momentum = self.normalizeIndicators(self.momentum)

        # rolling standard deviation
        self.rstd = pd.Series(pd.rolling_std(df_prices, window = 2), name = 'rstd').dropna()
        self.rstd[self.rstd < 0.5] = 0.5
        self.rstd[self.rstd > 4] = 4
        self.discretize_rstd = self.normalizeIndicators(self.rstd)

        # rolling rsi
        self.rsi = idc.get_rsi(df_prices, self.window_size).dropna()
        self.rsi[self.rsi < 40] = 40
        self.rsi[self.rsi > 70] = 70
        self.discretize_rsi = self.normalizeIndicators(self.rsi)

        self.discretize_holding = {0: 0, -1000: 1, 1000: 2}
        self.n_states = int(3 * 10**4)

    def discretizeIndicators(self):
        """
        Discretize indicators in 10 bins labelized from 0 to 9 @useless now
        """
        self.discretize_bband = pd.qcut(self.bband, q = 10, retbins=False, labels=False, duplicates = 'drop')
        self.discretize_momentum = pd.qcut(self.momentum, q = 10, retbins=False, labels=False, duplicates = 'drop')
        self.discretize_rstd = pd.qcut(self.rstd, q = 10, retbins=False, labels=False, duplicates = 'drop')
        self.discretize_rsi = pd.qcut(self.rsi, q = 10, retbins=False, labels=False, duplicates = 'drop')
        self.discretize_holding = {0: 0, -1000: 1, 1000: 2}

    
    def buildState(self, day):
        """
        Given a trading day, build the corresponding state using discretized indicators
        using the following format : holding*10^4 * rstd*10^3 + bband*10^2 + mmtum*10^1 + rsi*10^0
        @params day: the corresponding trading day index
        """
        state = self.discretize_holding[self.net_holdings] * 10**4 + \
        self.discretize_bband[day] * 10**3 +\
        self.discretize_momentum[day]*10**2 +\
        self.discretize_rsi[day]*10**1 +\
        self.discretize_rstd[day]*10**0 
        

        return int(state)
    
    def doAction(self, action, day, df_prices):
        """
        Do an action as a trading action and return a reward based on daily_returns
        """
        # trade action
        trade_value = self.getTradeValue(signal_value = self.signal_values[action])
        self.net_holdings += trade_value
        # compute rewards
        daily_ret = (df_prices.iloc[day] / df_prices.iloc[day-1] - 1) * (1 + self.impact) * self.net_holdings
        return daily_ret, trade_value
    
    def getTradeValue(self, signal_value):
        """
        Return a long or short or buy or sell value depending on the net holding and signal value
        """
        trade_value = 0
        if (self.net_holdings == 0):
            # BUY or SELL
            trade_value = signal_value
        elif ((signal_value + self.net_holdings) == 0):
            # LONG or SHORT
            trade_value = 2 * signal_value
        return trade_value
    
    def addEvidence(self, symbol = "JPM", sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 10000): 
        """
        Add price data (Evidence) to train the learner
        @params symbol : the asset's symbol
        @params sd : start date
        @params df: end date
        @params sv: portfolio start_value
        """
        # get symbol prices df
        df_prices_all = get_data([symbol], pd.date_range(sd, ed))
        df_prices = df_prices_all[symbol]
        prices_SPY = df_prices_all['SPY']  # only SPY, for comparison later            
        # initialization trades df
        
        self.df_trades = pd.DataFrame(data = np.zeros(len(df_prices)),\
                                 index = df_prices.index, columns=['trades'])
          
        # build indicators
        self.buildIndicators(df_prices, symbol = symbol)
        
        # new QLearner
        self.learner = ql.QLearner(num_states=self.n_states,\
            num_actions = self.n_actions, \
            alpha = self.alpha, \
            gamma = self.gamma, \
            rar = self.rar, \
            radr = self.radr, \
            dyna = 0, \
            verbose=False)

        # train learner
        previous_reward = 0
        repeated_tot_reward = 0
        initial_day = self.window_size
        
        for epoch in range(1, self.max_epoch):
            self.net_holdings = 0
            day_index = df_prices.index[initial_day]
            state = self.buildState(day_index)
            action = self.learner.querysetstate(state)
            total_reward = 0

            for day in range(initial_day + 1 , len(self.discretize_bband)):
                day_index = df_prices.index[day]
                # new state
                state_prime = self.buildState(day_index)
                reward, _ = self.doAction(action, day, df_prices)
                action = self.learner.query(state_prime, reward)
                total_reward += reward

            if self.verbose: print("Epoch {}, cumulative reward = {} ".format(epoch, total_reward))
            # convergence after 10 same tot_reward
            if previous_reward == total_reward:
                repeated_tot_reward +=1
            else:
                repeated_tot_reward = 0

            if repeated_tot_reward > 10:
                break
            previous_reward = total_reward
        
        if self.verbose: print("it better be a DataFrame", type(self.df_trades)) # it better be a DataFrame!
        

    def testPolicy(self, symbol = "JPM", \
        sd = dt.datetime(2010,1,1), ed =dt.datetime(2011,12,31), \
        sv = 10000):
        
        # get symbol prices df
        df_prices_all = get_data([symbol], pd.date_range(sd, ed))
        df_prices = df_prices_all[symbol]
        prices_SPY = df_prices_all['SPY']  # only SPY, for comparison later
        #if self.verbose: print(df_prices)
        
        # initialization
        total_reward = 0
        self.df_trades = pd.DataFrame(data = np.zeros(len(df_prices)),\
                                 index = df_prices.index, columns=[symbol])

        # build indicators
        self.net_holdings = 0
        self.buildIndicators(df_prices, symbol = symbol)
        initial_day = self.window_size
        
        for day in range(initial_day, len(self.discretize_bband)):
            day_index = df_prices.index[day]
            # new state 
            state = self.buildState(day_index)
            action = self.learner.querysetstate(state)
            reward, trade = self.doAction(action, day, df_prices)
            # update df trades
            self.df_trades.iloc[day] = trade
            total_reward += reward   
            
        if self.verbose: print(type(self.df_trades)) # it better be a DataFrame!
        if self.verbose: print(self.df_trades)
        if self.verbose: print(df_prices_all)
        return self.df_trades

if __name__=="__main__":
    print("One does not simply think up a strategy")

#author adurocher3 Alexis DUROCHER