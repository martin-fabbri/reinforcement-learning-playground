"""
Simple Reinforcement Learning agent resolving a deterministic problem
"""
import numpy as np
import pandas as pd
import time


np.random.seed(2)  # reproducible results

# Environment config
N_STATES = 6  # length of the one-dimensional world
ACTIONS = ['left', 'right']  # available actions
FRESH_TIME = 0.3

# Hyperparameters
epsilon = 0.9  # greedy policy
alpha = 0.1  # learning rate
gamma = 0.9  # discount rate
MAX_EPISODES = 10  # maximum episodes in a episode


def create_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions
    )
    return table


def update_environment(s, episode, step_counter):
    env_list = ['-'] * (N_STATES -1) + ['T']
    if s == "terminal":
        print("terminal")
    else:
        env_list[s] = "o"
        interaction = "".join(env_list)
        print(f"\r{interaction}")
        time.sleep(FRESH_TIME)


def learn(): 
    q = create_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        s = 0
        done = False
        update_environment(s, episode, step_counter)

if __name__ == "__main__":
    learn()
