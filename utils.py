import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

'''
Tinker Source #1:

a) which indicators to use
b) time ranges for indicators
'''

technical_indicators = [
	('sma', 15), 
	('sma', 60), 
	('ema', 30),
	('rsi', 14), 
	('atr', 14),
	('roc', 10),
	('macd', [12, 26, 9]), #returns macd, macdsignal, macdhist in talib
	('stoch', [21, 7, 7]) #returns slowk, slowd in talib
	#more indicators here
]


def get_data(col='close'):
    """ Returns a n x n_step array """
    stock_values = []
    for stock_csv in os.listdir('data/'):
        stock_data = pd.read_csv('data/{}'.format(stock_csv))
        curr_stock_value = stock_data.values[::-1]
	
	inputs = {
    	    'open': curr_stock_value[:, 0], 
	    'high': curr_stock_value[:, 1],
	    'low': curr_stock_value[:, 2],
	    'close': curr_stock_value[:, 3],
	    'volume': curr_stock_value[:, 4]
	}
 
	for indicator in technical_indicators:
		indicator_data = None
		if(indicator[0] == 'macd'):
			macd, macdsignal, macdhist = Function(indicator[0])(inputs['close'], 
				fastperiod=indicator[1][0], 
				slowperiod=indicator[1][1], 
				signalperiod=indicator[1][2])
					

	#just get the close column - TEMP
	stock_values.append(curr_stock_value[:, 3])
	

    # recent price are at top; reverse it
    return np.array(stock_values)


def get_scaler(env):
    """ Takes a env and returns a scaler for its observation space """
    low = [0] * (env.n_stock * 2 + 1)

    high = []
    max_price = env.stock_price_history.max(axis=1)
    min_price = env.stock_price_history.min(axis=1)
    max_cash = env.init_invest * 3  # 3 is a magic number...
    max_stock_owned = max_cash // min_price
    for i in max_stock_owned:
        high.append(i)
    for i in max_price:
        high.append(i)
    high.append(max_cash)

    scaler = StandardScaler()
    scaler.fit([low, high])
    print("Scaler is {}".format(scaler))
    return scaler


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
