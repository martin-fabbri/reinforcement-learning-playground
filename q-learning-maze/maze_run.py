import numpy as np
from maze_env import MazeEnv, UP, DOWN, LEFT, RIGHT
from maze_q_learning import QLearning

MAX_EPISODES = 100
NUM_STATES = 16

ACTIONS = np.array([UP, DOWN, RIGHT, LEFT])


def update():
    for episode in range(MAX_EPISODES):
        # initial observation
        observation = env.reset()
        done = False

        while not done:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = rl.choose_action(observation)

            # RL take action and get next observation
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            rl.learn(observation, action, reward, observation_, done)

            observation = observation_
    print("---------- Q -------------")
    print(rl.q)

if __name__ == "__main__":
    env = MazeEnv()
    rl = QLearning(NUM_STATES, ACTIONS)
    env.board.after(100, update)
    env.board.mainloop()