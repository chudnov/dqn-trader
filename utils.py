import os
import pandas as pd
import numpy as np
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


def get_data(stock_symbol, is_detrend=False):
    """ Returns a n_step array """
    # Data frame w/ open, close, high, low, volume values and reverse
    df = pd.read_csv('data/{}'.format(stock_symbol + ".csv")).iloc[::-1]

    if(is_detrend):
        df = detrend(df)

    # Convert to stockdataframe
    stock = Sdf.retype(df)

    for indicator in technical_indicators:
        stock.get(indicator)

    mask = main_features + technical_indicators
    stock_table = stock.dropna(
        how='any').loc[:, mask].to_numpy()
    return stock_table


def get_split_data(stock_symbol, ratio, detrend):
    data = np.around(get_data(stock_symbol, detrend))

    data_size = data.shape[0]
    end_row_train = (int)(data_size * (ratio / 100))
    end_row_validate = (data_size - end_row_train)//2 + end_row_train

    data_split = {}
    data_split["train"] = data[:end_row_train]
    data_split["validation"] = data[end_row_train:end_row_validate]
    data_split["test"] = data[end_row_validate:]
    return data_split

def fit(data_split, mode):	
    if(mode == 'train'):
        scaler = MinMaxScaler((0.1, 1))
        data_split[mode] = scaler.fit_transform(data_split[args.mode])
        # save scaler to disk
        with open('scalers/{}-{}.p'.format(timestamp, mode), 'wb') as fp:
            pickle.dump(scaler, fp)
            print("Saved scaler in {}".format(fp))
    else:
        # load scaler
        scaler = pickle.load(open(args.scaler, 'rb'))
	data_split[mode] = scaler.fit_transform(data_split[mode])

def detrend(df):
    del df[df.columns[0]]
    new_df = df.diff(periods=1).iloc[1:]
    new_df = new_df.add(abs(new_df.min()))
    return new_df


def view_signals(prices, signals):
    df = pd.DataFrame()
    s = np.array(signals).flatten()
    df['Close'] = prices.flatten()
    df['Buy'] = pd.Series(np.where(s == 2, 1, 0))
    df['Sell'] = pd.Series(np.where(s == 0, 1, 0))
    plt.figure(figsize=(20, 5))
    plt.plot(df['Close'], zorder=0)
    plt.scatter(df[df['Buy'] == 1].index.tolist(), df.loc[df['Buy'] ==
                                                          1, 'Close'].values, zorder=1, label='skitscat', color='green', s=30, marker=".")

    plt.scatter(df[df['Sell'] == 1].index.tolist(), df.loc[df['Sell'] ==
                                                           1, 'Close'].values, zorder=1, label='skitscat', color='red', s=30, marker=".")
    plt.xlabel('Timestep')
    plt.ylabel('Close Price')
    plt.show()


def maybe_make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
