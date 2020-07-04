import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
from empyrical import sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio
import math

class TradingEnv(gym.Env):
    """
    A 1-stock trading environment.

    State: [indicators for stock price, cash in hand, sharpe ratio]
    
    Action: sell (0), hold (1), and buy (2)
      - when selling, sell all the shares
      - when buying, buy as many as cash in hand allows
      - if buying multiple stock, equally distribute cash in hand and then utilize the balance
    """

    def __init__(self, train_data, init_invest, reward_len, reward_func, slippage_rate=0.001):
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
        self.stock_borrowed = None
        self.indicators = None
        self.returns = None
        self.current_position = None
        #self.is_short = None
        self.reward_len = reward_len
        self.reward_func = reward_func
 
        self.slippage_rate = slippage_rate

        # Trading statistics
        # number of completed trades
        self.trade_count = None
        # Number of profitable completed trades
        self.trades_profitable = None
        # Balance from completed trades
        self.cash_in_hand = None
 
        # action space
        self.action_space = 3
        # state space
        self.observation_space = self.stock_indicators_history.shape[1] 

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
            'risk_ratio': self._risk_adj(),
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
        self.stock_borrowed = 0
        self.stock_owned = 0
        self.stock_price = self.stock_price_history[self.cur_step]
        self.indicators = self.stock_indicators_history[self.cur_step, :]
        self.cash_in_hand = self.init_invest
        self.current_position = 1 
        self.trade_count = 0
        self.trades_profitable = 0
	#self.is_short = False

        return self._get_obs()

    def _step(self, action):
        self.returns.append(self._get_val()) 
        self._trade(action)
        self.cur_step += 1
        # update price
        self.stock_price = self.stock_price_history[self.cur_step]
        # update indicators
        self.indicators = self.stock_indicators_history[self.cur_step, :]
        reward = self._reward() 
        #if(self.is_short): reward = -reward
        done = self.cur_step == self.n_step - 1
        return self._get_obs(), reward, done

    def _get_obs(self):
        obs = []
        obs.extend(list(self.indicators))
        return obs

    def _risk_adj(self):
        tmp = self.reward_len
        self.reward_len = self.n_step
        r = round(self._reward(), 2)
        self.reward_len = tmp
        return r

    def _reward(self):
        length = min(self.cur_step, self.reward_len)
        returns = np.diff(self.returns)[-length:]

        if self.reward_func == 'sortino':
          reward = sortino_ratio(returns)
        elif self.reward_func == 'calmar':
          reward = calmar_ratio(returns)
        elif self.reward_func == 'omega':
          reward = omega_ratio(returns)
        else:
           reward = sharpe_ratio(returns) 

        return reward if abs(reward) != math.inf and not np.isnan(reward) else 0

    def _get_val(self):
        return self.stock_owned * self.stock_price + self.cash_in_hand #if not self.is_short else self.cash_in_hand - self.stock_borrowed * self.stock_price

    def _trade(self, action):
        #print("Action {} with {} cash and {} stock owned \n".format(action, self.cash_in_hand, self.stock_owned))       
 
        # hold
        if(action == 1):
            pass
        
        # sell
        elif(action == 0):
            self.cash_in_hand += (1-self.slippage_rate) * self.stock_price * self.stock_owned 
            curr_profit = self.stock_price - self.enter_price 
            self.enter_price = 0
            self.stock_owned = 0
            
            if(curr_profit > 0):
                self.trades_profitable += 1
        
        # buy
        else:
            self.trade_count += 1                       
            num_to_purchase = self.cash_in_hand // ((1 + self.slippage_rate) * self.stock_price)
            self.stock_owned += num_to_purchase
            self.cash_in_hand -= num_to_purchase * self.stock_price
            self.enter_price = self.stock_price
        
        self.current_position = action
        self.signals.append(action)
