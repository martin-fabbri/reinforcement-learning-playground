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
EPSILON = 0.9  # greedy policy
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount rate
MAX_EPISODES = 10  # maximum episodes in a episode

def create_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions
    )
    return table


def render(s, episode, step_counter):
    env_list = ['-'] * (N_STATES -1) + ['T']
    if s == "terminal":
        interaction = 'Episode %s: total_steps = %s' % (
        episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[s] = "o"
        interaction = "".join(env_list)
        print(f"\r{step_counter} {interaction}")
        time.sleep(FRESH_TIME)


def reset():
    return 0


def draw():
    return np.random.choice(ACTIONS)


def choose_action(s, q_table):
    state_actions = q_table.iloc[s, :]
    if np.random.uniform() > EPSILON or state_actions.all() == 0:
        action_name = draw()  # act non-greedy or state-action
    else:
        action_name = state_actions.argmax()  # act greedy
    return action_name


def step(action, state):
    """Run one timestep of the environment's dynamic
    :param action: an action provided by the environment
    :return: (observation, reward, done)
    """
    observation = state
    reward = 0
    done = False
    if action == "right":
        if state == N_STATES - 2:
            observation = "terminal"
            reward = 1
            done = True
        else:
            observation = state + 1
    else:
        # move left
        if state != 0:
            observation = state - 1
    return observation, reward, done


def learn():
    q = create_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = reset()
        done = False
        render(state, episode, step_counter)
        while not done:
            action = choose_action(state, q)
            observation, reward, done = step(action, state)
            if observation != "terminal":
                q_target = reward + GAMMA * q.iloc[observation, :].max()
            else:
                q_target = reward  # next state is terminal
            q_predict = q.ix[state, action]
            q.ix[state, action] += ALPHA * (q_target - q_predict)
            state = observation
            render(state, episode, step_counter)
            step_counter += 1
    return q

if __name__ == "__main__":
    q = learn()
    print('\r\nQ-table:\n')
    print(q)
