import numpy as np


def learn(S, A, env, discount):
    v = np.zeros(len(S))
    new_v = np.zeros(len(S))
    pi = np.ones((len(S), len(A))) / len(A)
    delta = 1
    epsilon = 1e-4
    log = []

    while delta > epsilon:
        delta = 0
        for s in S:
            best_q = -1 << 20
            best_a = 0
            for a in A:
                q = 0
                for s_next in S:
                    q += env["p"][s, a, s_next] * \
                        (env["r"][s, a, s_next] + discount * v[s_next])

                if q > best_q:
                    best_a = a
                    best_q = q

            pi[s] = np.zeros(len(A))
            pi[s][best_a] = 1
            delta += abs(best_q - v[s])
            v[s] = best_q

        log.append(delta)

    return v, pi, log
