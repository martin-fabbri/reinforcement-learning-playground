import numpy as np
import pandas as pd


class QLearning(object):
    def __init__(self, num_states, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.np_random = np.random.RandomState()
        self.np_random.seed(5)
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.num_states = num_states
        data = np.zeros((self.num_states, len(actions)))
        self.q = pd.DataFrame(data, columns=self.actions)
        print(self.q)

    def choose_action(self, observation):
        if np.random.uniform() < self.epsilon:
            state_action = self.q.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.argmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, done):
        q_predict = self.q.ix[s, a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * self.q.ix[s_, :].max()  # next state is not terminal

        self.q.ix[s, a] += self.lr * (q_target - q_predict)  # update
