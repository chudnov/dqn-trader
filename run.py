import pickle
import time
import numpy as np
import argparse
import re
from stockstats import StockDataFrame as Sdf
import matplotlib.pyplot as plt
import pandas as pd

from envs import TradingEnv
from agent import DQNAgent
from utils import fit, get_split_data, maybe_make_dir, view_signals

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episode', type=int, default=2000,
                        help='number of episode to run')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch size for experience replay')
    parser.add_argument('-i', '--initial_invest', type=int, default=20000,
                        help='initial investment amount')
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='"train", "validate", or "test"', choices=['train', 'validate', 'test'])
    parser.add_argument('-w', '--weights', type=str,
                        help='a trained model weights')
    parser.add_argument('-s', '--scaler', type=str,
                        help='scaler data')
    parser.add_argument('-r', '--ratio', type=int, default=70,
                        help='% of data for train')
    parser.add_argument('--symbol', type=str,
                        help='symbol of stock')
    parser.add_argument('-l', '--layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('-n', '--neurons', type=int, default=24,
                        help='number of neurons layers')
    parser.add_argument('-d', '--detrend', type=bool, default=False,
                        help='detrend or not')
    parser.add_argument('--mem', type=int, default=2000,
                        help='memory size of agent')

    args = parser.parse_args()

    # Set up dirs
    maybe_make_dir('weights')
    maybe_make_dir('portfolio_val')
    maybe_make_dir('scalers')

    # Get time. Same as when it was trained if we are validating/testing
    timestamp = time.strftime('%Y%m%d%H%M') if args.mode == 'train' else re.findall(
        r'\d{12}', args.weights)[0]

    # Get data split
    data_split = get_split_data(args.symbol, args.ratio, args.detrend)

    # Fit data
    fit(data_split, args.mode, timestamp)

    # Create environment
    env = TradingEnv(data_split[args.mode], args.initial_invest)

    # Create agent
    agent = DQNAgent(env.observation_space, env.action_space,
                     args.layers, args.neurons, args.mem, batch_size=args.batch_size, epsilon=1 if args.mode == "train" else 0)

    # Store portfolio value after iterations
    portfolio_value = [args.initial_invest]

    # Load weights if not training
    if args.mode != 'train':
        agent.load(args.weights)

    # Warm up the agent
    if(args.mode == 'train'):
        state = env.reset()
        for _ in range(args.mem):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)

    for e in range(args.episode):
        state = env.reset()
        for time in range(env.n_step):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            if args.mode == 'train':
                agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                count, ratio, bal, unrealized = env._stats().values()
                print("episode: {}/{}, performed {} trades, {:,.2%} are profitable, with a final episode unrealized pnl of: ${:,.2f}".format(
                    e + 1, args.episode, count, ratio, unrealized))
                # append episode end portfolio value
                portfolio_value.append(unrealized)
                break
            if args.mode == 'train' and len(agent.memory) > args.batch_size:
                agent.replay()
        if args.mode == 'train' and (e + 1) % 10 == 0:  # checkpoint weights
            agent.save('weights/{}-dqn.h5'.format(timestamp))

    # save portfolio value history to disk
    with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
        pickle.dump(portfolio_value, fp)
        print("Saved episode values in {}".format(fp))

    if(args.mode != 'train'):
        view_signals(data_split[args.mode], env.signals)
