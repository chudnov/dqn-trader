import pickle
import numpy as np
import argparse
import re
import matplotlib.pyplot as plt
import pandas as pd
from envs import TradingEnv
from agent import DQNAgent
from model import mlp
from utils import fit, get_split_data, maybe_make_dir, view_signals

np.set_printoptions(threshold=np.inf)


# Meta
ratio = 70
initial_investment = 20000
episodes = 500

# Model 
activation = 'relu'
loss = 'mse' 
learning_rate = 1e-2
dqn_type = 1

# Agent
mem = 2000
update_freq = 10
batch_size = 64
gamma = 0.99
epsilon = 1
epsilon_min = 0.01
epsilon_start = 30000
epsilon_decay = 0.997  

# Env
reward_func = 'sharpe'
window_size = 1 
slippage_rate = 0.001

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='"train" or "test"', choices=['train', 'test'])
    parser.add_argument('--symbol', type=str,
                        help='symbol of stock')
    parser.add_argument('-d', '--detrend', type=bool, default=False,
                        help='detrend or not')
   
    args = parser.parse_args()

    # Set up dirs
    maybe_make_dir('weights')
    maybe_make_dir('portfolio_val')
    maybe_make_dir('scalers')

    # Get data split
    data_split = get_split_data(args.symbol, ratio, args.detrend)

    # Fit data
    fit(data_split, args.mode, args.symbol)

    # Create environment
    env = TradingEnv(data_split[args.mode], initial_investment, window_size, reward_func, slippage_rate)

    # Create model
    model = mlp(env.observation_space, env.action_space, activation, loss, learning_rate, dqn_type)

    # Create agent
    agent = DQNAgent(env, args.mode, model, mem, update_freq, batch_size, gamma, epsilon, epsilon_min, epsilon_start, epsilon_decay) 

    # Store portfolio value after iterations
    portfolio_value = [initial_investment]

    # Load weights if not training
    if args.mode != 'train':
        episodes = 1
        agent.load('weights/{}-weights.h5'.format(args.symbol))

    # Warm up the agent
    '''
    if(args.mode == 'train'):
        state = env.reset()
        for _ in range(mem):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
    '''

    for e in range(episodes):
        state = env.reset()
        while (True):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            if args.mode == 'train':
                agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                count, ratio, risk_return, unrealized = env._stats().values()
                print("episode: {}/{}, performed {} trades, {:,.2%} of one's sold are profitable, with a risk-adjusted ratio of {} and final episode unrealized pnl of: ${:,.2f}".format(
                    e + 1, episodes, count, ratio, risk_return, unrealized))
                # append episode end portfolio value
                portfolio_value.append(unrealized)
                break
            if args.mode == 'train' and len(agent.memory) > batch_size:
                agent.replay()
        if args.mode == 'train' and (e + 1) % 10 == 0:  # checkpoint weights
            agent.save('weights/{}-weights.h5'.format(args.symbol))

    # save portfolio value history to disk
    with open('portfolio_val/{}-returns.p'.format(args.symbol), 'wb') as fp:
        pickle.dump(portfolio_value, fp)
        print("Saved episode values in {}".format(fp))

    if(args.mode != 'train'):
        view_signals(data_split[args.mode], env.signals)
