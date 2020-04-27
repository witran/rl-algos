import numpy as np


def move(r, c, n_row, n_col, direction):
    if direction == 0:
        return max(r - 1, 0), c
    elif direction == 1:
        return r, min(c + 1, n_col - 1)
    elif direction == 2:
        return min(r + 1, n_row - 1), c
    else:
        return r, max(c - 1, 0)


def create_env(n_row, n_col, grid_reward, terminals):
    n_states = n_row * n_col
    n_actions = 4

    p = np.zeros((n_states, n_actions, n_states))
    r = np.zeros((n_states, n_actions, n_states))

    for row in range(n_row):
        for col in range(n_col):
            s = row * n_col + col
            for a in range(n_actions):
                next_r, next_c = move(row, col, n_row, n_col, a)
                next_s = next_r * n_col + next_c
                r[s, a, next_s] = grid_reward[next_s]
                p[s, a, next_s] = 1

    for (row, col) in terminals:
        s = row * n_col + col
        for a in range(n_actions):
            r[s, a] = np.zeros(n_states)
            p[s, a] = np.zeros(n_states)
            p[s, a, s] = 1

    return {"p": p, "r": r}
