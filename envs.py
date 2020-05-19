import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools


class TradingEnv(gym.Env):
    """
    A x-stock trading environment.

    State: [current stock prices, indicators for stock price, cash in hand]
      - array of length n_stock * (num indicators + 1) + 1
      - price is discretized (to integer) to reduce state space
      - use close price for each stock
      - cash in hand is evaluated at each step based on action performed

    Action: sell (0), hold (1), and buy (2)
      - when selling, sell all the shares
      - when buying, buy as many as cash in hand allows
      - if buying multiple stock, equally distribute cash in hand and then utilize the balance
    """

    def __init__(self, train_data, init_invest=20000):
        # data
        self.stock_price_history = train_data[:, :, 0]
        self.stock_indicators_history = train_data[:, :, 2:]
        self.n_stock, self.n_step = self.stock_price_history.shape

        indicators_max = self.stock_indicators_history.max(axis=1)
        indicators_min = self.stock_indicators_history.min(axis=1)

        self.signals = None

        # instance attributes
        self.init_invest = init_invest
        self.cur_step = None
        self.enter_price = None
        self.stock_price = None
        self.stock_owned = None
        self.indicators = None

        # Trading statistics
        # number of completed trades
        self.trade_count = None
        # Number of profitable completed trades
        self.trades_profitable = None
        # Balance from completed trades
        self.cash_in_hand = None
        # Balance from completed and open trades
        self.account_balance_unrealised = None

        # action space
        self.action_space = spaces.Discrete(3**self.n_stock)

        # observation space: give estimates in order to sample and build scaler
        stock_max_price = self.stock_price_history.max(axis=1)
        price_range = [[0, mx] for mx in stock_max_price]
        indicator_range = [[list(indicators_min[i])[j], list(indicators_max[i])[j]] for i in range(
            0, len(indicators_max)) for j in range(len(list(indicators_min[i])))]
        cash_in_hand_range = [[0, init_invest * 2]]
        self.observation_space = spaces.MultiDiscrete(
            price_range + indicator_range + cash_in_hand_range)

        # seed and start
        self._seed()
        self._reset()

    def _stats(self):
        '''
        Returns a dict of trading statistics
        '''
        if self.trade_count == 0:
            win_loss_ratio = 0
        else:
            win_loss_ratio = self.trades_profitable / self.trade_count

        return {
            'trade_count': self.trade_count,
            'win_loss_ratio': win_loss_ratio,
            'account_balance': self.cash_in_hand,
            'unrealised_pl': self._get_val(),
        }

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.signals = []
        self.cur_step = 0
        self.enter_price = [0] * self.n_stock
        self.stock_owned = [0] * self.n_stock
        self.stock_price = self.stock_price_history[:, self.cur_step]
        self.indicators = self.stock_indicators_history[:, self.cur_step, :]
        self.cash_in_hand = self.init_invest

        self.trade_count = 0
        self.trades_profitable = 0
        self.account_balance_unrealised = self.init_invest

        return self._get_obs()

    def _step(self, action):
        assert self.action_space.contains(action)
        prev_val = self._get_val()  # self.cash_in_hand for realized pnl
        self.cur_step += 1
        # update price
        self.stock_price = self.stock_price_history[:, self.cur_step]
        # update indicators
        self.indicators = self.stock_indicators_history[:, self.cur_step, :]
        self._trade(action)
        cur_val = self._get_val()  # self.cash_in_hand for realized pnl
        reward = cur_val - prev_val
        done = self.cur_step == self.n_step - 1
        return self._get_obs(), reward, done

    def _get_obs(self):
        obs = []
        obs.extend(list(self.stock_price))
        obs.extend(list(self.indicators.flatten()))
        obs.append(self.cash_in_hand)
        return obs

    def _get_val(self):
        return np.sum(self.stock_owned * self.stock_price) + self.cash_in_hand

    def _trade(self, action):
        # all combo to sell(0), hold(1), or buy(2) stocks
        action_combo = list(itertools.product([0, 1, 2], repeat=self.n_stock))
        action_vec = action_combo[action]

        # one pass to get sell/buy index
        sell_index = []
        buy_index = []
        for i, a in enumerate(action_vec):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # two passes: sell first, then buy; might be naive in real-world settings
        if sell_index:
            for i in sell_index:
                if(self.stock_owned[i] == 0):
                    continue

                ##BADD##
                self.signals.append(0)
                self.trade_count += 1
                self.cash_in_hand += self.stock_price[i] * self.stock_owned[i]
                profit = self.stock_price[i] - self.enter_price[i]
                if profit > 0:
                    self.trades_profitable += 1
                self.enter_price[i] = 0
                self.stock_owned[i] = 0

        if buy_index:
            can_buy = True

            ##BADD##
            self.signals.append(2)

            while can_buy:
                for i in buy_index:
                    if self.cash_in_hand > self.stock_price[i]:
                        self.stock_owned[i] += 1  # buy one share
                        self.cash_in_hand -= self.stock_price[i]
                        self.enter_price[i] = self.stock_price[i]
                    else:
                        can_buy = False
            ##BADD##
            return
        ##BADD##
        self.signals.append(1)
