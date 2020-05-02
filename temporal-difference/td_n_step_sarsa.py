import numpy as np
import environment

''' some implementation notes
    1-step -> 1 reward term + 
    n-step -> n reward term
        last reward term will have discount: gamma ^ (n - 1)
        q term will have discount: gamma ^ n

    delta = g + discounted q

    maintain 2 running vars
        t: runs from len(actions) - 1
        t_update: runs from t -> 0
        inclusive range [t, t_update] length always equal trace_length
'''

BOOTSTRAP_EXPECTED = "expected"
BOOTSTRAP_SARSA = "sarsa"
BOOTSTRAP_Q = "q"


def learn(S, A, env, agent):
    pi = np.ones((len(S), len(A))) / len(A)
    q = np.ones((len(S), len(A)))
    is_policy_updated = True

    discount = agent["discount"] or 0.9
    step_size = agent["step_size"] or 0.1
    epsilon = agent["epsilon"] or 0.1
    trace_length = agent["trace_length"] or 1
    bootstrap = agent["bootstrap"] or BOOTSTRAP_SARSA
    n_iterations = agent["n_iterations"] or 500
    learn_online = agent["learn_online"]
    plan_background = agent["plan_background"]
    if not agent["learn_online"] and not agent["plan_background"]:
        raise Exception("learn_online and plan_background both False")

    # logging
    step = 0
    visit_count = np.zeros(len(S))
    error_history = []
    length_history = []

    while step < n_iterations:
        error = 0
        if learn_online:
            history, error = get_sample_and_learn_online(
                env, q, bootstrap, discount, step_size, epsilon)
        else:
            history = get_sample(env, q, epsilon)

        if plan_background:
            error += backup(q, pi, bootstrap, discount, step_size,
                            epsilon, trace_length, history)

        new_pi = greedy_policy(q)
        pi = new_pi

        error_history.append(error)
        length_history.append(len(history[0]))

        if n_iterations - step < 20:
            for s in history[0]:
                visit_count[s] += 1

        step += 1

    return q, pi, (error_history, length_history, visit_count)


def get_sample(env, q, epsilon):
    s = env["start_state"]
    h_s = [s]
    h_a = []
    h_r = []
    over = False
    step = 0
    max_step = env["n_row"] * env["n_col"] * 3

    while not over and step < max_step:
        step += 1
        a = epsilon_greedy_select(q[s], epsilon)
        s, r, over = environment.step(env, s, a)
        h_a.append(a)
        h_r.append(r)
        h_s.append(s)
        if step == max_step:
            h_r[len(h_r) - 1] = -1000

    return h_s, h_a, h_r


def get_sample_and_learn_online(env, q, bootstrap, discount, step_size, epsilon):
    s = env["start_state"]
    h_s = [s]
    h_a = []
    h_r = []
    over = False
    step = 0
    error = 0

    while not over:
        step += 1
        a = epsilon_greedy_select(q[s], epsilon)
        s_next, r, over = environment.step(env, s, a)

        if not over:
            if bootstrap == BOOTSTRAP_SARSA:
                a_next = epsilon_greedy_select(q[s_next], epsilon)
                q_next = q[s_next, a_next]
            elif bootstrap == BOOTSTRAP_EXPECTED:
                q_next = expected_q(q[s_next], epsilon)
            elif bootstrap == BOOTSTRAP_Q:
                q_next = max(q[s_next])
        else:
            q_next = 0

        delta = r + discount * q_next - q[s, a]
        q[s, a] = q[s, a] + step_size * delta

        s = s_next

        h_a.append(a)
        h_r.append(r)
        h_s.append(s)

        error += abs(step_size * delta)

    return (h_s, h_a, h_r), error


def epsilon_greedy_select(action_values, epsilon):
    if np.random.random() > epsilon:
        max_value = np.max(action_values)
        choices = [a for a in range(
            len(action_values)) if action_values[a] == max_value]
        return np.random.choice(choices)
    else:
        return np.random.randint(len(action_values))


def backup(q, pi, bootstrap, discount,
           step_size, epsilon, trace_length, history):
    s, a, r = history
    t = len(a) - 1
    t_update = t
    g = 0
    error = 0

    while t_update >= 0:
        # update g
        g = g * discount + r[t_update]

        # compute delta term
        if t == len(a) - 1:
            q_last = 0
        elif bootstrap == BOOTSTRAP_SARSA:
            q_last = q[s[t + 1], a[t + 1]]
        elif bootstrap == BOOTSTRAP_EXPECTED:
            q_last = expected_q(q[s[t + 1]], epsilon)
        elif bootstrap == BOOTSTRAP_Q:
            q_last = max(q[s[t + 1]])

        s_up = s[t_update]
        a_up = a[t_update]
        delta = g + (discount ** (t - t_update)) * q_last - q[s_up, a_up]

        # perform update
        q[s_up, a_up] += step_size * delta
        error += abs(step_size * delta)

        # slide the window
        # the inclusive range [t, t_update] maintains max length of trace_length
        t_update -= 1
        if t - t_update + 1 == trace_length:
            g = g - r[t] * (discount ** (trace_length - 1))
            t -= 1

    return error


def expected_q(action_values, epsilon):
    return (1 - epsilon) * max(action_values) + \
        np.sum(epsilon / len(action_values) * action_values)


def greedy_policy(q):
    new_pi = np.zeros(q.shape)
    for s in range(len(q)):
        new_pi[s][np.argmax(q[s])] = 1

    return new_pi
