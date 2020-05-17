import pickle
import time
import numpy as np
import argparse
import re

from envs import TradingEnv
from agent import DQNAgent
from utils import get_data, get_scaler, maybe_make_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--episode', type=int, default=2000,
                        help='number of episode to run')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch size for experience replay')
    parser.add_argument('-i', '--initial_invest', type=int, default=20000,
                        help='initial investment amount')
    parser.add_argument('-m', '--mode', type=str, required=True,
                        help='"train", "validate", or "test"')
    parser.add_argument('-w', '--weights', type=str,
                        help='a trained model weights')
    parser.add_argument('-r', '--ratio', type=int, default=70,
                        help='% of data for train')
    parser.add_argument('-l', '--layers', type=int, default=2,
			help='number of hidden layers') 
    parser.add_argument('-n', '--neurons', type=int, default=24,
			help='number of neurons layers')
    args = parser.parse_args()

    if args.mode not in ['train', 'validate', 'test']:
        quit()

    maybe_make_dir('weights')
    maybe_make_dir('portfolio_val')

    timestamp = time.strftime('%Y%m%d%H%M')

    data = np.array([np.around(d) for d in get_data()])

    data_size = data[0].shape[0]
    end_row_train = (int)(data_size * (args.ratio / 100))
    end_row_validate = (data_size - end_row_train)//2 + end_row_train

    train_data = np.array([d[:end_row_train, :] for d in data])
    validation_data = np.array(
        [d[end_row_train:end_row_validate, :] for d in data])
    test_data = np.array([d[end_row_validate:, :] for d in data])

    env = TradingEnv(train_data, args.initial_invest)
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size,
                     args.layers, args.neurons)
    scaler = get_scaler(env)

    portfolio_value = []

    # Append initial account value
    portfolio_value.append(args.initial_invest)

    if args.mode != 'train':
        # remake the env with validation data
        env = TradingEnv(validation_data if args.mode ==
                         'validate' else test_data, args.initial_invest)
        # load trained weights
        agent.load(args.weights)
        # when validate, the timestamp is same as time when weights was trained
        timestamp = re.findall(r'\d{12}', args.weights)[0]

    for e in range(args.episode):
        state = env.reset()

        state = scaler.transform([state])

        for time in range(env.n_step):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = scaler.transform([next_state])
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
                agent.replay(args.batch_size)
        if args.mode == 'train' and (e + 1) % 10 == 0:  # checkpoint weights
            agent.save('weights/{}-dqn.h5'.format(timestamp))

    # save portfolio value history to disk
    with open('portfolio_val/{}-{}.p'.format(timestamp, args.mode), 'wb') as fp:
        pickle.dump(portfolio_value, fp)
        print("Saved episode values in {}".format(fp))
