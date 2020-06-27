import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import itertools
from utils import sharpe_ratio

class TradingEnv(gym.Env):
    """
    A 1-stock trading environment.

    State: [indicators for stock price, cash in hand, sharpe ratio]
    
    Action: sell (0), hold (1), and buy (2)
      - when selling, sell all the shares
      - when buying, buy as many as cash in hand allows
      - if buying multiple stock, equally distribute cash in hand and then utilize the balance
    """

    def __init__(self, train_data, init_invest, slippage_rate=0.001):
        # data
        self.stock_price_history = train_data[0]
        self.stock_indicators_history = train_data[1]
        self.n_step = self.stock_price_history.shape[0]

        self.signals = None

        # instance attributes
        self.init_invest = init_invest
        self.cur_step = None
        self.enter_price = None
        self.stock_price = None
        self.stock_owned = None
        self.indicators = None
        self.returns = None

        self.slippage_rate = slippage_rate

        # Trading statistics
        # number of completed trades
        self.trade_count = None
        # Number of profitable completed trades
        self.trades_profitable = None
        # Balance from completed trades
        self.cash_in_hand = None
        # Total profit
        self.profit = None
        # Balance from completed and open trades
        self.account_balance_unrealised = None
 
        # action space
        self.action_space = 3
        # state space
        self.observation_space = self.stock_indicators_history.shape[1] + 2

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
            'sharpe_ratio': sharpe_ratio(self.returns),
            'unrealised_pl': self._get_val(),
        }

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self.signals = []
        self.returns = []
        self.cur_step = 0
        self.enter_price = 0
        self.stock_owned = 0
        self.stock_price = self.stock_price_history[self.cur_step]
        self.indicators = self.stock_indicators_history[self.cur_step, :]
        self.cash_in_hand = self.init_invest
        self.profit = 0

        self.trade_count = 0
        self.trades_profitable = 0
        self.account_balance_unrealised = self.init_invest

        return self._get_obs()

    def _step(self, action):
        prev_val = self._get_val()
        self._trade(action)
        self.cur_step += 1
        # update price
        self.stock_price = self.stock_price_history[self.cur_step]
        # update indicators
        self.indicators = self.stock_indicators_history[self.cur_step, :]
        cur_val = self._get_val()  
        reward = cur_val - prev_val
        self.returns.append(reward) 
        done = self.cur_step == self.n_step - 1
        return self._get_obs(), reward, done

    def _get_obs(self):
        obs = []
        obs.extend(list(self.indicators))
        obs.append(self.cash_in_hand)
        obs.append(sharpe_ratio(self.returns))
        return obs

    def _get_val(self):
        return self.stock_owned * self.stock_price + self.cash_in_hand

    def _trade(self, action):
        # sell
        if(action == 0):
            if(self.stock_owned == 0):
                self.signals.append(1)
                return
            self.trade_count += 1
            
            self.cash_in_hand += self.stock_price * self.stock_owned
            curr_profit = self.stock_price - self.enter_price
            self.profit += curr_profit * self.stock_owned
            
            #self.cash_in_hand += self.stock_price
            #curr_profit = self.stock_price - self.enter_price
            #self.profit += curr_profit
 
            if(curr_profit > 0):
                self.trades_profitable += 1
            self.enter_price = 0
            self.stock_owned = 0
            #self.stock_owned -= 1

        # buy
        elif(action == 2):
            if(self.cash_in_hand < self.stock_price):
                self.signals.append(1)
                return
            #self.trade_count += 1
            
            self.enter_price = self.stock_price
            num_to_purchase = self.cash_in_hand // self.stock_price
            self.stock_owned += num_to_purchase
            self.cash_in_hand -= num_to_purchase * self.stock_price
            
            #self.stock_owned += 1
            #self.cash_in_hand -= self.stock_price

        self.signals.append(action)
