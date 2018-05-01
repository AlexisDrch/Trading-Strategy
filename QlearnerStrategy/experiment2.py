
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
Experiment 2: Provide an hypothesis regarding how changing the value of impact should affect in 
sample trading behavior and results (provide at least two metrics). Conduct an experiment with 
JPM on the in sample period to test that hypothesis. 
Provide charts, graphs or tables that illustrate the results of your experiment.
The code that implements this experiment and generates the relevant charts and 
data should be submitted as experiment2.py
"""

def run_experiment2(symbol="SINE_FAST_NOISE", sd_in =dt.datetime(2008,1,1), ed_in =dt.datetime(2009,12,31), \
	sd_out =dt.datetime(2010,1,1), ed_out =dt.datetime(2011,12,31), sv=100000):
	# note : to compare apple to apple : need to change window size in the strategy learned to fit 
	# with the manual strategy : bband = 5, mmtm = 10, rstd = 20, rsi = 30
	
	impact_range = np.arange(0, 0.1, 0.001)
	stats = pd.Series(np.zeros(4), index=['cr', 'adr', 'sddr', 'sr'], name='0')
	for impact in impact_range:

		learner = sl.StrategyLearner(verbose = False, impact = impact) # constructor
		learner.addEvidence(symbol = symbol, sd= sd_in, ed = ed_in, sv = sv) # training phase
		df_trades = learner.testPolicy(symbol = symbol, sd = sd_out, ed=ed_out, sv = sv) # testing phase
		
		df_portval = normalize_dataframe(compute_portvals(df_trades, impact= impact))
		
		df_buy_signals= df_trades[df_trades.values >= 1000]
		df_sell_signals = df_trades[df_trades.values <= -1000]

		df_prices_all = get_data([symbol], pd.date_range(sd_out, ed_out))
		df_prices = df_prices_all[symbol]
		benchmark = normalize_dataframe(get_benchmark(df_prices))
		stats_porfolio = compute_portfolio_stats(df_portval.values , name = impact)
		stats = pd.concat((stats, stats_porfolio), axis = 1)

	fig = plt.figure()
	ax = fig.add_subplot(111, title = 'cumulative return and sharpe ratio')
	ax2 = ax.twinx()
	stats.T.fillna(method='bfill', inplace = True)
	stats.T.cr.plot(ax = ax, c = 'b', legend = 'cr')
	stats.T.sr.plot(ax = ax2, c = 'r', legend = 'sharpe ratio')
	ax.set_xlabel('impact')
	ax.set_ylabel('Cumulative return', color = 'b')
	ax2.set_ylabel('sharpe ratio', color ='r')
	plt.grid(True)
	plt.legend()
	plt.savefig('./output/impact-influ-cr-sr.png')
	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(111, title = 'average and standard deviation daily return ')
	stats.T.fillna(method='bfill', inplace = True)
	stats.T.adr.plot(ax = ax, c = 'g', legend = 'adr')
	stats.T.sddr.plot(ax = ax, c = 'orange', legend = 'sddr')
	ax.set_xlabel('impact')
	plt.grid(True)
	plt.legend(loc='best')
	plt.savefig('./output/impact-influ-ar-sddr.png')
	plt.show()


run_experiment2(symbol="JPM", sd_out = dt.datetime(2008,1,1), ed_out =dt.datetime(2009,12,31))

#author adurocher3 Alexis DUROCHER