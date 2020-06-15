from copy import deepcopy
from datetime import datetime
import gym
import dqn
import constants
import numpy as np
from numpy import save, load
from matplotlib import pyplot as plt

env = gym.make('CartPole-v0')
# env = gym.make('LunarLander-v2')
# env = gym.make('MountainCar-v0')

agent = dqn.Agent(
    discount=0.99,
    softmax_temperature=0.001,

    greedy_epsilon_max=0.2,
    greedy_epsilon_min=0.0001,
    # ~300_000 steps needed -> cap at ~120_000
    # greedy_max_step=180_000,
    greedy_max_step=12_000,

    policy=constants.POLICY_EPSILON,
    bootstrap_type=constants.BOOTSTRAP_MAX_Q,
    use_double_q=False,

    # store config
    priority_epsilon=0,
    priority_alpha=0,
    store_size=50_000,

    # main loop
    n_steps=500_000,
    n_episodes=50,
    target_update_interval=1,
    n_steps_to_start_training=1000,

    # logging config
    log=False,
    log_interval=1000,
    demo=False,
    demo_interval=10,
    smooth=10,

    # env config
    timeout=1000,
    timeout_reward=0,

    # model connfig
    encode_time=False,
    # loss=constants.LOSS_HUBER,
    loss=constants.LOSS_MSE,
    # optimizer=constants.OPTIMIZER_ADAM,
    optimizer=constants.OPTIMIZER_RMSPROP,
    rmsprop_lr=0.001,
    adam_lr=1e-3,
    adam_beta_m=0.9,
    adam_beta_v=0.999,
    adam_epsilon=1e-8,
    # n_batches=1,
    # batch_size=32,
    # slow converge but stable result
    # n_batches=8,
    # batch_size=8,
    # fast converge but unstable - 18 53
    n_batches=4,
    batch_size=8,
    epochs=1,
    # 18 58
    # epochs=2,
)

# TODO - proper param study here
# generator for params, draw average over few runs


def generate_params():
    param_options = {
        "n_batches": [1, 2, 4],
        "batch_size": [8, 16, 32],
        "rmsprop_lr": [0.0001, 0.00025, 0.0005, 0.001]
    }

    # names = ["n_batches", "batch_size", "rmsprop_lr"]
    names = ["rmsprop_lr"]
    combinations = []
    current_combination = {}

    def generate(i):
        if i == len(names):
            combinations.append(deepcopy(current_combination))
            return

        for option in param_options[names[i]]:
            current_combination[names[i]] = option
            generate(i + 1)

    generate(0)

    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    print('file-id', now)
    for i, combination in enumerate(combinations):
        print(i, combination)

    color = [np.random.rand(3,) for _ in combinations]
    print(color)

    fig, ax = plt.subplots(1)
    n_runs = 3

    for i, combination in enumerate(combinations):
        # add param to job queue
        # create param
        for param in combination:
            agent[param] = combination

        qnet, reward_histories, loss_histories = dqn.run(
            env, agent, n_runs=5, show_plot=False)

        # block wait for result
        rh = np.array(reward_histories)
        mu = rh.mean(axis=0)
        sigma = rh.std(axis=0)
        t = np.arange(rh.shape[1])
        ax.plot(t, mu, lw=2, label='combination-' + str(i), color=color[i])
        ax.fill_between(t, mu + sigma, mu - sigma,
                        facecolor=color[i], alpha=0.2)

        # save data
        # torch.save(qnet.state_dict(), 'models/model_' + now)
        # save('data/reward_histories_' + now, reward_histories)
        # save('data/loss_histories_' + now, loss_histories)

    ax.legend(loc='upper left')
    ax.set_title('params study - reward over time')
    ax.set_xlabel('episodes')
    ax.set_ylabel('reward')
    ax.grid()
    plt.savefig('figures/exp_' + now)
    plt.show()


generate_params()
