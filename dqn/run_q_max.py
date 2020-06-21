from copy import deepcopy
from datetime import datetime
import multiprocessing as mp
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt
import gym

import dqn
import constants
import executor

N_RUNS = 10
ENV_CODE = "CartPole-v0"
# ENV_CODE = "LunarLander-v2"
# ENV_CODE = "MountainCar-v0"


def main():
    agents = generate_params()
    pprint(agents)
    result = execute(agents)
    plot(result)


def generate_params():
    agent = dqn.Agent(
        discount=0.99,
        softmax_temperature=0.001,

        greedy_epsilon_max=0.2,
        greedy_epsilon_min=0.0001,
        # ~300_000 steps needed -> cap at ~120_000
        # greedy_max_step=180_000,
        # for cartpole, saturate at 24k -> use 12k max step
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
        n_episodes=80,
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

    param_options = {
        "n_batches": [1, 2, 4],
        "batch_size": [8, 16, 32],
        # "rmsprop_lr": [0.0001, 0.00025, 0.0005, 0.001]
        # "rmsprop_lr": [0.001, 0.0012, 0.0015]
        "rmsprop_lr": [0.0015, 0.0012, 0.001]
    }

    # names = ["n_batches", "batch_size", "rmsprop_lr"]
    names = ["rmsprop_lr"]
    combinations = []
    current_combination = {}
    agents = []

    def generate(i):
        if i == len(names):
            combinations.append(deepcopy(current_combination))
            return

        for option in param_options[names[i]]:
            current_combination[names[i]] = option
            generate(i + 1)

    generate(0)

    for i, combination in enumerate(combinations):
        for param in combination:
            setattr(agent, param, combination[param])
        agents.append(deepcopy(agent))
        print(i, combination)

    return agents


def execute(agents):
    launch_args = []
    n_agents = len(agents)
    n_runs = N_RUNS
    env_code = ENV_CODE
    n_episodes = getattr(agents[0], "n_episodes")

    for i in range(n_agents):
        for j in range(n_runs):
            task_id = i * n_runs + j
            # launch n_runs times
            launch_args.append((task_id, env_code, agents[i]))

    # print(launch_args)
    # return

    result = executor.launch(
        run, launch_args, output_size=n_episodes, n_workers=8)

    print('here')

    # average result to find mean, std
    result = result.reshape((n_agents, n_runs, n_episodes))
    print(result)
    return result


def run(worker_id, monitor, args):
    task_id, env_code, agent = args
    env = gym.make(env_code)
    _, reward_history, _ = dqn.learn(
        env, agent, monitor=monitor, worker_id=worker_id)
    env.close()
    return reward_history


def plot(rh):
    # plot result
    fig, ax = plt.subplots(1)

    # mean & std across runs of each agent
    mu = rh.mean(axis=1)
    sigma = rh.std(axis=1)

    # t = n_episodes
    t = np.arange(rh.shape[2])

    color = [np.random.rand(3,) for _ in range(rh.shape[0])]

    for i in range(rh.shape[0]):
        ax.plot(t, mu[i], lw=2, label='combination-' + str(i), color=color[i])
        ax.fill_between(t, mu[i] + sigma[i], mu[i] - sigma[i],
                        facecolor=color[i], alpha=0.2)

    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    # print('file-id', now)
    # save data
    # torch.save(qnet.state_dict(), 'models/model_' + now)
    # np.save('data/reward_histories_' + now, reward_histories)
    # np.save('data/loss_histories_' + now, loss_histories)

    ax.legend(loc='upper left')
    ax.set_title('params study - reward over time')
    ax.set_xlabel('episodes')
    ax.set_ylabel('reward')
    ax.grid()
    plt.savefig('figures/exp_' + now)
    plt.show()


main()
