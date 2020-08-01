import gym
#from gym import spaces
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

    def __init__(self, train_data, init_invest, window_size, reward_func, slippage_rate):
        # data
        self.stock_price_history = train_data[0]
        self.stock_indicators_history = train_data[1]
        self.n_step = self.stock_price_history.shape[0]
        
        self.base_sharpe = None 
        self.signals = None

        # instance attributes
        self.init_invest = init_invest
        self.cur_step = None
        self.enter_price = None
        self.stock_price = None
        self.stock_owned = None
        self.stock_borrowed = None
        self.returns = None
        self.current_position = None
        #self.is_short = None
        self.reward_func = reward_func
        self.window_size = window_size
 
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
        self.observation_space = (self.window_size, self.stock_indicators_history.shape[1])

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
        self.base_sharpe = [self.init_invest] * self.window_size 
        self.cur_step = self.window_size
        self.enter_price = 0
        self.stock_borrowed = 0
        self.stock_owned = 0
        self.stock_price = self.stock_price_history[self.cur_step]
        self.cash_in_hand = self.init_invest
        self.current_position = 1 
        self.trade_count = 0
        self.trades_profitable = 0
	#self.is_short = False

        return self._get_obs()

    def _step(self, action):
        self.returns.append(self._get_val())
        reward = self._reward_pnl(action)
        self._trade(action)
        self.cur_step += 1
        self.stock_price = self.stock_price_history[self.cur_step]
        #print("Action {} with reward {}. Price was {} and now is {}".format(action, reward, self.stock_price_history[self.cur_step - 1], self.stock_price_history[self.cur_step]))
        #if(self.is_short): reward = -reward
        done = self.cur_step == self.n_step - 1
        return self._get_obs(), reward, done

    def _get_obs(self):
        obs = []
        #obs = self.stock_indicators_history[self.cur_step-self.window_size:self.cur_step, :]
        obs = self.stock_indicators_history[self.cur_step, :]
        return obs
	
    def _risk_adj(self):
        returns = np.diff(self.returns)

        if self.reward_func == 'sortino':
          reward = sortino_ratio(returns)
        elif self.reward_func == 'calmar':
          reward = calmar_ratio(returns)
        elif self.reward_func == 'omega':
          reward = omega_ratio(returns)
        else:
          reward = sharpe_ratio(returns)

        return round(reward, 2) if abs(reward) != math.inf and not np.isnan(reward) else 0

    def _reward_pnl(self, action):
        prev = self.stock_price_history[self.cur_step] if action > 0 else self.enter_price
        cur = self.stock_price_history[self.cur_step + 1] if action > 0 else self.stock_price
        reward = math.log(cur/prev)
        return reward

    def _reward(self):
        sharpe_hist = pd.DataFrame(self.base_sharpe).pct_change().fillna(0.0).values
        A = np.mean(sharpe_hist)
        B = np.mean(sharpe_hist**2)
        delta_A = sharpe_hist[self.cur_step-1] - A
        delta_B = sharpe_hist[self.cur_step-1]**2 - B
        Dt = (B*delta_A - 0.5*A*delta_B) / (B-A**2)**(3/2)
        return round(Dt[0], 3)
 
    def _get_val(self):
        return self.stock_owned * self.stock_price + self.cash_in_hand #if not self.is_short else self.cash_in_hand - self.stock_borrowed * self.stock_price

    def _trade(self, action):
        #print("{} cash and {} stock owned \n".format(self.cash_in_hand, self.stock_owned))       
 
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
