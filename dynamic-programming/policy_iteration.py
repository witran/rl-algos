import numpy as np


def learn(S, A, env, discount):
    is_policy_updated = True

    pi = np.ones((len(S), len(A))) / len(A)
    v = np.zeros((len(S)))
    log = []

    while is_policy_updated:
        v, total_delta = evaluate_policy(
            S, A, v, pi, env["p"], env["r"], discount)
        
        pi, is_policy_updated = update_policy(
            S, A, v, pi, env["p"], env["r"], discount)

        log.append(total_delta)

    return v, pi, log


def evaluate_policy(S, A, v, pi, p, r, discount):
    epsilon = 1e-4
    delta = 1
    total_delta = 0
    new_v = np.zeros(len(S))
    current_v = np.zeros(len(S))
    
    while delta > epsilon:
        delta = 0
        for s in S:
            new_v[s] = 0
            for a in A:
                for s_next in S:
                    new_v[s] += pi[s, a] * p[s, a, s_next] * \
                        (r[s, a, s_next] + discount * current_v[s_next])

            delta = max(delta, abs(new_v[s] - current_v[s]))
            total_delta += abs(new_v[s] - current_v[s])
            current_v[s] = new_v[s]

    return new_v, total_delta


def update_policy(S, A, v, pi, p, r, discount):
    is_policy_updated = False
    new_pi = np.zeros((len(S), len(A)))

    for s in S:
        best_a = 0
        best_q = -1 << 20
        for a in A:
            q = 0
            for s_prime in S:
                q += p[s, a, s_prime] * \
                    (r[s, a, s_prime] + discount * v[s_prime])

            if q > best_q:
                best_q = q
                best_a = a

        new_pi[s] = np.zeros((len(A)))
        new_pi[s, best_a] = 1

        if not np.allclose(new_pi[s], pi[s]):
            is_policy_updated = True

    return new_pi, is_policy_updated
