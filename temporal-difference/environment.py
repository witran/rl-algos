import numpy as np


def create(**kwargs):
    keys = ["step_reward", "cliff_reward", "grid"]

    start_state = None
    terminal_states = []
    cliff_states = []
    grid = kwargs["grid"]
    n_row = len(grid)
    n_col = len(grid[0])

    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == "S":
                start_state = r * n_col + c
            elif grid[r][c] == "T":
                terminal_states.append(r * n_col + c)
            elif grid[r][c] == "C":
                cliff_states.append(r * n_col + c)

    assert(all(map(lambda key: key in kwargs, keys)))

    env = {
        "n_row": n_row,
        "n_col": n_col,
        "step_reward": kwargs["step_reward"],
        "cliff_reward": kwargs["cliff_reward"],
        "start_state": start_state,
        "terminal_states": terminal_states,
        "cliff_states": cliff_states
    }
    return env


def step(env, state, action):
    start_state = env["start_state"]
    terminal_states = env["terminal_states"]
    cliff_states = env["cliff_states"]
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

    if next_state in cliff_states:
        next_state = start_state
        reward = cliff_reward

    if next_state in terminal_states:
        reward = 0
        over = True

    return next_state, reward, over
