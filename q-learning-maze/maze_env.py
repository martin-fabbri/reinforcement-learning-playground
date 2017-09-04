"""
Maze Env

State               Description             Reward
-----               -----------             ------
h                   blackhole (fail)        -1
t                   treasure (success)      +1
g                   ground                  0
"""
import numpy as np
import time
from maze_board_ui import MazeBoard, MAZE_H, MAZE_W

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

MAP = np.array([
    ["g", "g", "g", "g"],
    ["g", "g", "h", "g"],
    ["g", "h", "t", "g"],
    ["g", "g", "g", "g"]
])


class MazeEnv:
    """Has the following members
    - nS: Number of states
    - nA: Number of actions
    - P: transitions
    """
    def __init__(self):
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.board_row = 0
        self.board_column = 0
        self.board = MazeBoard()

    def step(self, a):
        if a == UP and self.board_row != 0:
            self.board_row -= 1
            self.board.step(a)
        elif a == DOWN and self.board_row != 3:
            self.board_row += 1
            self.board.step(a)
        elif a == LEFT and self.board_column != 0:
            self.board_column -= 1
            self.board.step(a)
        elif a == RIGHT and self.board_column != 3:
            self.board_column += 1
            self.board.step(a)

        state_ = self.board_row * MAZE_H + self.board_column

        done = False
        reward = 0

        cell_type_ = MAP[self.board_row, self.board_column]
        if cell_type_ == "t":
            reward = 1
            done = True
        elif cell_type_ == "h":
            reward = -1
            done = True

        return state_, reward, done

    def reset(self):
        self.board_row = 0
        self.board_column = 0
        self.board.reset()
        return 0

    def render(self):
        time.sleep(0.1)
        self.board.update()

