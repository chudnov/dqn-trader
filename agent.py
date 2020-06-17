from collections import deque
import random
import numpy as np
from model import mlp


class DQNAgent(object):
    """ A simple Deep Q agent """

    def __init__(self, state_size, action_size, num_layers, num_neurons, memory_size, update_target_freq=100, batch_size=64, gamma=0.95, epsilon=1.0, epsilon_min=.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_target_freq = update_target_freq
        self.batch_size = batch_size
        self.step = 0
        self.model = mlp(state_size, action_size, num_layers, num_neurons)
        self.model_sub = mlp(state_size, action_size, num_layers, num_neurons)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0])  # returns action

    def update_target_model(self):
        self.model_sub.set_weights(self.model.get_weights())

    def replay(self):
        self.step += 1

        """ vectorized implementation; 30x speed up compared with for loop """
        minibatch = np.array(random.sample(self.memory, self.batch_size))

        states = np.concatenate(minibatch[:, 0]).reshape(
            self.batch_size, self.state_size)
        actions = np.array([tup[1] for tup in minibatch])
        rewards = np.array([tup[2] for tup in minibatch])
        next_states = np.concatenate(minibatch[:, 3]).reshape(
            self.batch_size, self.state_size)
        done = np.array([tup[4] for tup in minibatch])

        '''
        print("states are {}".format(states))
        print("actions are {}".format(actions))
        print("rewards are {}".format(rewards))
        print("next states are {}".format(next_states))
        print("done is {}".format(done))
        '''

        # Q(s', a)
        target = rewards + self.gamma * self.model_sub.predict(next_states)[range(
            self.batch_size), np.argmax(self.model.predict(next_states), axis=1)]
        # end state target is reward itself (no lookahead)
        target[done] = rewards[done]

        # Q(s, a)
        target_f = self.model.predict(states)
        # make the agent to approximately map the current state to future discounted reward
        target_f[range(self.batch_size), actions] = target

        if(self.step % self.update_target_freq == 0):
            self.update_target_model()

        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
