import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from stockstats import StockDataFrame as Sdf
import matplotlib.pyplot as plt

# Remove chaining warning
pd.options.mode.chained_assignment = None
# Remove summary printing
pd.options.display.max_rows = None


'''
Tinker Source #1:

a) which features to use
b) time ranges for indicators
'''

main_features = [
    'close',
    'volume'
]

# Check https://pypi.org/project/stockstats/ for more indicators
technical_indicators = [
    'rsi_14',
    'cci_14',
    'dx_14'
]


def get_data(is_detrend=False):
    """ Returns a n x n_step array """
    stock_values = []
    for stock_csv in os.listdir('data/'):
        if(stock_csv.startswith('.')):
            continue
        # Data frame w/ open, close, high, low, volume values and reverse
        df = pd.read_csv('data/{}'.format(stock_csv)).iloc[::-1]

        if(is_detrend):
            df = detrend(df)

        # Convert to stockdataframe
        stock = Sdf.retype(df)

        for indicator in technical_indicators:
            stock.get(indicator)

        mask = main_features + technical_indicators
        stock_table_with_indicators = stock.dropna(
            how='any').loc[:, mask].to_numpy()
        stock_values.append(stock_table_with_indicators)

    return np.array(stock_values)


'''
Tinker Source #2:

a) which scaler to use
'''


def get_scaler(env, max_profit_factor):
    """ Takes a env and returns a scaler for its observation space """

    low = []
    high = []

    indicators_max = env.stock_indicators_history.max(axis=1)
    indicators_min = env.stock_indicators_history.min(axis=1)

    max_cash = env.init_invest * max_profit_factor

    for i in range(0, len(indicators_max)):
        low.extend(list(indicators_min[i]))
        high.extend(list(indicators_max[i]))

    low.append(-env.init_invest)
    high.append(max_cash)

    scaler = MinMaxScaler((0.1, 1)) #or RobustScaler
    scaler.fit([low, high])

    print("Scaler is {}".format(scaler))
    return scaler


def detrend(df):
    del df[df.columns[0]]
    new_df = df.diff(periods=1).iloc[1:]
    new_df = new_df.add(abs(new_df.min()))
    return new_df


def view_signals(prices, signals):
    df = pd.DataFrame()
    s = np.array(signals).flatten()
    df['Close'] = prices[:, :, 0].flatten()
    df['Buy'] = pd.Series(np.where(s == 2, 1, 0))
    df['Sell'] = pd.Series(np.where(s == 0, 1, 0))
    plt.figure(figsize=(20,5))
    plt.plot(df['Close'], zorder=0)
    plt.scatter(df[df['Buy'] == 1].index.tolist(), df.loc[df['Buy'] ==
                                                       1, 'Close'].values, zorder=1, label='skitscat', color='green', s=30, marker=".")
    
    plt.scatter(df[df['Sell'] == 1].index.tolist(), df.loc[df['Sell'] ==
                                                       1, 'Close'].values, zorder=1,label='skitscat', color='red', s=30, marker=".")
    plt.xlabel('Timestep')  
    plt.ylabel('Close Price') 
    plt.show()


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
