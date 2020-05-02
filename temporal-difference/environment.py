import numpy as np


def to_cell(state, n_col):
    return state // n_col, state % n_col


def create(**kwargs):
    keys = ["step_reward", "cliff_reward", "grid"]

    start_cell = None
    terminal_cells = []
    cliff_cells = []
    grid = kwargs["grid"]
    n_row = len(grid)
    n_col = len(grid[0])

    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == "S":
                start_cell = (r, c)
            elif grid[r][c] == "T":
                terminal_cells.append((r, c))
            elif grid[r][c] == "C":
                cliff_cells.append((r, c))

    assert(all(map(lambda key: key in kwargs, keys)))

    env = {
        "start_state": start_cell[0] * n_col + start_cell[1],
        "n_row": n_row,
        "n_col": n_col,
        "step_reward": kwargs["step_reward"],
        "cliff_reward": kwargs["cliff_reward"],
        "start_cell": start_cell,
        "terminal_cells": terminal_cells,
        "cliff_cells": cliff_cells
    }
    return env


def step(env, state, action):
    start_cell = env["start_cell"]
    terminal_cells = env["terminal_cells"]
    cliff_cells = env["cliff_cells"]
    n_row = env["n_row"]
    n_col = env["n_col"]
    reward = env["step_reward"]
    cliff_reward = env["cliff_reward"]
    r = state // n_col
    c = state % n_col
    over = False
    next_state = None

    if action == 0:
        next_state = max(r - 1, 0) * n_col + c
    elif action == 1:
        next_state = r * n_col + min(c + 1, n_col - 1)
    elif action == 2:
        next_state = min(r + 1, n_row - 1) * n_col + c
    else:
        next_state = r * n_col + max(c - 1, 0)

    if (r, c) in cliff_cells:
        next_state = start_cell[0] * n_col + start_cell[1]
        reward = cliff_reward

    if (r, c) in terminal_cells:
        over = True

    return next_state, reward, over
