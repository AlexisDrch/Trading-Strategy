
import random
import util as ut
import numpy as np
import pandas as pd
import datetime as dt
import indicators as idc
import QLearner as ql
import matplotlib.pyplot as plt
from util import get_data, plot_data
import StrategyLearner as sl
from marketsimcode import *

"""
Experiment 1: Using exactly the same indicators that you used in manual_strategy, 
compare your manual strategy with your learning strategy in sample. 
Plot the performance of both strategies in sample along with the benchmark. 
Trade only the symbol JPM for this evaluation. 
The code that implements this experiment and generates the relevant charts 
and data should be submitted as experiment1.py
Describe your experiment in detail: Assumptions, parameter values and so on.
Describe the outcome of your experiment.
Would you expect this relative result every time with in-sample data? Explain why or why not.

"""


def run_experiment(symbol="SINE_FAST_NOISE", sd_in =dt.datetime(2008,1,1), ed_in =dt.datetime(2009,12,31), \
	sd_out =dt.datetime(2010,1,1), ed_out =dt.datetime(2011,12,31), sv=100000):
	# note : to compare apple to apple : need to change window size in the strategy learned to fit 
	# with the manual strategy : bband = 5, mmtm = 10, rstd = 20, rsi = 30
	learner = sl.StrategyLearner(verbose = False, impact = 0.000) # constructor
	learner.addEvidence(symbol = symbol, sd= sd_in, ed = ed_in, sv = sv) # training phase
	df_trades = learner.testPolicy(symbol = symbol, sd = sd_out, ed=ed_out, sv = sv) # testing phase
	
	df_portval = normalize_dataframe(compute_portvals(df_trades))
	
	df_buy_signals= df_trades[df_trades.values >= 1000]
	df_sell_signals = df_trades[df_trades.values <= -1000]

	df_prices_all = get_data([symbol], pd.date_range(sd_out, ed_out))
	df_prices = df_prices_all[symbol]
	benchmark = normalize_dataframe(get_benchmark(df_prices))
	stats_porfolio = compute_portfolio_stats(df_portval.values , name = 'qlearner strat')
	print(stats_porfolio.head())
	plt = idc.plot_strategy(df_prices, df_portval, benchmark, None, str('exp 1 - ' + symbol), df_buy_signals, df_sell_signals)
	plt.savefig('./output/experiment1_'+symbol+'.png')
	plt.show()


run_experiment(symbol="JPM", sd_out = dt.datetime(2008,1,1),ed_out =dt.datetime(2009,12,31))

#author adurocher3 Alexis DUROCHER
