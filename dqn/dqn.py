import math
import pprint
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
import gym
import numpy as np
import torch
from torch import nn
from torch import optim
import constants


class Agent():
    def __init__(self, **kwargs):
        # env config
        self.timeout = kwargs.get("timeout", 1000)
        self.timeout_reward = kwargs.get("timeout_reward", 0)
        self.discount = kwargs.get("discount", 0.99)
        self.state_size = None
        self.action_size = None

        # actor config
        self.policy = kwargs.get("policy", constants.POLICY_EPSILON)
        self.softmax_temperature = kwargs.get("softmax_temperature", 1.0)
        self.greedy_epsilon_max = kwargs.get("greedy_epsilon_max", 0.1)
        self.greedy_epsilon_min = kwargs.get("greedy_epsilon_min")
        self.greedy_max_step = kwargs.get("greedy_max_step", 100_000)

        # store config
        self.store_size = kwargs.get("store_size")
        self.priority_epsilon = kwargs.get("priority_epsilon")
        self.priority_alpha = kwargs.get("priority_alpha")

        # learner td config
        self.bootstrap_type = kwargs.get("bootstrap_type")
        self.use_double_q = kwargs.get("use_double_q")
        self.use_n_step = False

        # model config
        self.encode_time = kwargs.get("encode_time")
        self.loss = kwargs.get("loss")
        self.optimizer = kwargs.get("optimizer")
        self.rmsprop_lr = kwargs.get("rmsprop_lr")
        self.adam_lr = kwargs.get("adam_lr")
        self.adam_beta_m = kwargs.get("adam_beta_m")
        self.adam_beta_v = kwargs.get("adam_beta_v")
        self.adam_epsilon = kwargs.get("adam_epsilon")
        self.n_batches = kwargs.get("n_batches")
        self.batch_size = kwargs.get("batch_size")
        self.epochs = kwargs.get("epochs")

        # main loop config
        self.n_steps = kwargs.get("n_steps")
        self.n_episodes = kwargs.get("n_episodes")
        self.target_update_interval = kwargs.get("target_update_interval")
        self.n_steps_to_start_training = kwargs.get(
            "n_steps_to_start_training")
        self.smooth = kwargs.get("smooth", 10)

        # logging config
        self.log = kwargs.get("log")
        self.log_interval = kwargs.get("log_interval")
        self.demo = kwargs.get("demo")
        self.demo_interval = kwargs.get("demo_interval")


class Store():
    def __init__(self, max_size=1e6):
        self.buffer = []
        self.max_size = max_size
        self.rand_generator = np.random.RandomState(1)
        self.tail = 0

    def add(self, item, priority):
        if len(self.buffer) < self.max_size:
            self.buffer.append(item)
        else:
            self.buffer[self.tail] = item

        self.tail = (self.tail + 1) % self.max_size

    def sample(self, batch_size):
        idxs = self.rand_generator.choice(
            np.arange(len(self.buffer)), size=batch_size)

        s, a, r, s_next, done = [], [], [], [], []

        for idx in idxs:
            item = self.buffer[idx]
            s.append(item[0])
            a.append(item[1])
            r.append(item[2])
            s_next.append(item[3])
            done.append(item[4])

        s, r, s_next, done = map(lambda arr: torch.tensor(
            arr).float(), (s, r, s_next, done))
        a = torch.tensor(a)

        return s, a, r, s_next, done


# def run(env_code, agent, n_runs=1, show_plot=False):
#     # pprint.pprint(agent.__dict__)

#     qnet = None
#     reward_histories = []
#     loss_histories = []

#     env = gym.make(env_code)

#     for _ in range(n_runs):
#         qnet, reward_history, loss_history = learn(env, agent)
#         reward_histories.append(reward_history)
#         # loss_histories.append(loss_history)

#     env.close()
#     # return qnet, reward_histories, loss_histories
#     return reward_histories


# def plot(title, data, file_id, save=False, show=False):
#     for d in data:
#         plt.plot(d)
#     plt.title(title)
#     if save:
#         plt.savefig('figures/' + title + '_' + file_id)
#     if show:
#         plt.show()


def model(input_size, output_size):
    hidden_size = 256
    # hidden_size = 64
    l1 = nn.Linear(input_size, hidden_size)
    l2 = nn.Linear(hidden_size, output_size)
    # torch.nn.init.orthogonal_(l1.weight)
    # torch.nn.init.zeros_(l1.bias)
    # torch.nn.init.orthogonal_(l2.weight)
    # torch.nn.init.zeros_(l2.bias)
    return nn.Sequential(l1, nn.ReLU(), l2)


def encode(s, step, agent):
    if agent.encode_time:
        return np.concatenate((s, [step / agent.timeout]))
    return s


def learn(env, agent, monitor=None, worker_id=None):
    # model
    state_size = env.observation_space.shape[0]
    if agent.encode_time:
        state_size += 1
    action_size = env.action_space.n
    agent.state_size = state_size
    agent.action_size = action_size

    # print("input:{}, output:{}".format(state_size, action_size))

    qnet = model(state_size, action_size)
    qnet_target = deepcopy(qnet)

    if agent.optimizer == constants.OPTIMIZER_ADAM:
        optimizer = optim.Adam(
            qnet.parameters(),
            lr=agent.adam_lr,
            betas=(
                agent.adam_beta_m,
                agent.adam_beta_v),
            eps=agent.adam_epsilon)
    elif agent.optimizer == constants.OPTIMIZER_RMSPROP:
        optimizer = optim.RMSprop(qnet.parameters(), lr=agent.rmsprop_lr)

    # store
    store = Store(agent.store_size)

    # debugging data
    r_sum = 0
    r_sum_window = []
    r_sum_window_length = agent.smooth
    loss = 0
    reward_history = []
    loss_history = []

    # main loop
    step = 0
    n_episodes = 0
    s = encode(env.reset(), step, agent)

    # use_tqdm = not agent.log and not agent.demo
    use_tqdm = False
    progress = None
    if use_tqdm:
        progress = tqdm(total=agent.n_episodes)

    if monitor:
        monitor.update(worker_id, 0, agent.n_episodes)

    for i in range(agent.n_steps):
        # act
        a, s_next, r, done = act(env, agent, qnet, s, step)

        # store
        store.add((s, a, r, s_next, done), 0)

        # loop
        if done:
            n_episodes += 1
            monitor.update(worker_id, n_episodes, agent.n_episodes)
            r_sum += r
            if len(r_sum_window) >= r_sum_window_length:
                del r_sum_window[0]
            r_sum_window.append(r_sum)
            reward_history.append(sum(r_sum_window) / len(r_sum_window))

            if use_tqdm:
                progress.update(1)

            s = encode(env.reset(), step, agent)
            r_sum = 0
            step = 0

            if n_episodes == agent.n_episodes:
                if use_tqdm:
                    progress.close()
                break

        else:
            s = s_next
            r_sum += r
            step += 1

        # learn
        if i > agent.batch_size:
            # train
            loss = 0
            qnet_target = deepcopy(qnet)
            for _ in range(agent.n_batches):
                loss = train(qnet, qnet_target, optimizer, store, agent)
            loss_history.append(loss)

            # copy net every target update interval
            if (i + 1) % agent.target_update_interval == 0:
                qnet_target = deepcopy(qnet)

            # demo on interval
            if agent.demo and step == 0 and \
                    (n_episodes + 1) % agent.demo_interval == 0:
                print("----- DEMO -----")
                last_episode_reward = play(env, agent, qnet, render=True)
                print("last episode reward", last_episode_reward)
                print("----------------")

            # print debug on interval
            if agent.log and (i + 1) % agent.log_interval == 0:
                print("-----")
                print("step #{}, num episodes played: {}, store size: {} loss: {}, last {} episodes avg={} best={} worst={}".format(
                    i + 1,
                    n_episodes,
                    len(store.buffer),
                    round(loss, 4),
                    len(r_sum_window),
                    round(sum(r_sum_window) / len(r_sum_window), 4),
                    round(max(r_sum_window), 4),
                    round(min(r_sum_window), 4)
                ))

    return qnet, reward_history, loss_history


def act(env, agent, qnet, s, current_step):
    q = qnet(torch.tensor(s).float())
    if agent.policy == constants.POLICY_SOFTMAX:
        pi = policy_softmax(q, agent.softmax_temperature).detach().numpy()
    else:
        epsilon = get_epsilon(
            agent.greedy_epsilon_max,
            agent.greedy_epsilon_min,
            agent.greedy_max_step,
            current_step)
        pi = policy_epsilon(q, epsilon).detach().numpy()
    a = sample_action(pi)
    s_next, r, done, info = env.step(a)
    s_next = encode(s_next, current_step + 1, agent)

    if current_step + 1 > agent.timeout:
        done = True
        r = agent.timeout_reward

    return a, s_next, r, done


def play(env, agent, qnet, render=False):
    max_step = 1000
    step = 0
    done = False
    s = encode(env.reset(), 0, agent)
    r_sum = 0
    actions = []

    while not done and step < max_step:
        if render:
            env.render()
        q = qnet(torch.tensor(s).float())
        a = q.argmax().item()
        actions.append(a)
        s_next, r, done, info = env.step(a)
        s = encode(s_next, step, agent)
        step += 1
        r_sum += r

    if render:
        print(actions)

    s = env.reset()

    return r_sum


def sample_action(pi):
    return np.random.choice(np.arange(len(pi)), p=pi).item()


def policy_softmax(q_values, tau=1.):
    preferences = q_values / tau
    max_preference = preferences.max(dim=-1, keepdim=True)[0]
    numerator = (preferences - max_preference).exp()
    denominator = numerator.sum(dim=-1, keepdim=True)
    pi = (numerator / denominator).squeeze()
    return pi


# linearly decay until saturation
def get_epsilon(epsilon_max, epsilon_min, max_step, step):
    return epsilon_max + min(step, max_step) / max_step * (epsilon_min - epsilon_max)


def policy_epsilon(q_values, epsilon):
    pi = torch.ones(q_values.shape) * epsilon / q_values.shape[-1]
    max_q = torch.max(q_values, dim=-1, keepdim=True)[0]
    mask = (q_values == max_q).float()
    max_q_dist = mask * (1 - epsilon) / mask.sum(dim=-1, keepdim=True)
    pi += max_q_dist
    return pi


def get_q(qnet, s, a):
    q_values = qnet(s)
    indexes = torch.arange(q_values.shape[0]) * q_values.shape[1] + a
    q = q_values.take(indexes)
    return q


def get_q_target_expected_sarsa(qnet, qnet_target, discount, r, s_next, done, use_double_q):
    q_values_next = qnet_target(s_next)

    if use_double_q:
        pi = policy_softmax(qnet(s_next))
    else:
        pi = policy_softmax(q_values_next)

    bootstrap_term = (pi * q_values_next).sum(dim=-1) * (1 - done)
    return (r + discount * bootstrap_term)


def get_q_target_max_q(qnet, qnet_target, discount, r, s_next, done, use_double_q):
    q_values_next = qnet_target(s_next)
    if use_double_q:
        # argmax by q_a
        q_max_indexes = torch.arange(
            q_values_next.shape[0]) * q_values_next.shape[1] + torch.argmax(qnet(s_next), dim=-1)
        # q_b[argmax[q_a]]
        q_max = q_values_next.take(q_max_indexes)
    else:
        q_max = torch.max(q_values_next, dim=-1)[0]

    bootstrap_term = q_max * (1 - done)
    return (r + discount * bootstrap_term)


def train(qnet, qnet_target, optimizer, store, agent):
    discount = agent.discount
    batch_size = agent.batch_size
    epochs = agent.epochs
    use_double_q = agent.use_double_q
    state_size = agent.state_size

    if agent.loss == constants.LOSS_HUBER:
        loss_fn = torch.nn.SmoothL1Loss()
    elif agent.loss == constants.LOSS_MSE:
        loss_fn = torch.nn.MSELoss()
    else:
        raise Exception("Unknown loss function " + agent.loss)

    loss_sum = 0

    s, a, r, s_next, done = store.sample(batch_size)

    assert s.shape == s_next.shape == torch.Size(
        [batch_size, agent.state_size])
    assert a.shape == r.shape == done.shape == torch.Size([batch_size])

    for epoch in range(epochs):
        # q & q_target are computed only for the selected action
        # and have shape of (batch_size)
        q = get_q(qnet, s, a)

        if agent.bootstrap_type == constants.BOOTSTRAP_MAX_Q:
            q_target = get_q_target_max_q(
                qnet, qnet_target, discount, r, s_next, done, use_double_q)
        else:
            q_target = get_q_target_expected_sarsa(
                qnet, qnet_target, discount, r, s_next, done, use_double_q)

        loss = loss_fn(q, q_target)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        loss_sum += loss.item()

    return loss_sum


def test_get_epsilon():
    assert np.allclose(get_epsilon(0.2, 0.1, 100_000, 0), 0.2)
    assert np.allclose(get_epsilon(0.2, 0.1, 100_000, 25_000), 0.175)
    assert np.allclose(get_epsilon(0.2, 0.1, 100_000, 50_000), 0.15)
    assert np.allclose(get_epsilon(0.2, 0.1, 100_000, 75_000), 0.125)

    print('get_epsilon() passed')


def test_policy_epsilon():
    epsilon = 0.1
    q_values = torch.tensor([
        [1.5, 5.0, 5.0, 5.0],
        [0.9, 0.8, 0.7, 0.7]
    ])
    pi = policy_epsilon(q_values, epsilon)
    pi_expected = torch.tensor([
        [0.025, 0.325, 0.325, 0.325],
        [0.925, 0.025, 0.025, 0.025]
    ])
    assert torch.allclose(pi, pi_expected)

    q_values = torch.tensor([1.5, 5., 5., 5.])
    pi = policy_epsilon(q_values, epsilon)
    pi_expected = torch.tensor([0.025, 0.325, 0.325, 0.325])
    assert torch.allclose(pi, pi_expected)

    print('policy_epsilon() passed')


test_policy_epsilon()
test_get_epsilon()
