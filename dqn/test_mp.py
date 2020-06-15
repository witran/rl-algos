import multiprocessing as mp
import time
import random
import numpy as np
from atpbar import atpbar, register_reporter, find_reporter, flush

mp.set_start_method('fork', force=True)

# test if we even need mp.Array? we do, it has sharedmem
# try with numpy maybe? nope doesn't work
# TODO: atpbar
# add tracing for all worker's progress
# add tracing for total - inner outer probably


def run_task(result_table, worker_id, args):
    # execute task
    task_id, params = args
    n_episodes = params["n_episodes"]

    # write result
    # for i in range(n_episodes):
    for i in atpbar(range(n_episodes), name='worker:{}'.format(worker_id)):
        id_mean = task_id * n_episodes * 2 + i
        id_std = id_mean + n_episodes
        time.sleep(0.001)
        result_table[id_mean] = random.random()
        result_table[id_std] = random.random()
        # result_table[0, task_id * n_episodes + i] = float(task_id)
        # result_table[1, task_id * n_episodes + i] = float(task_id)

    flush()


def worker(worker_id, result_table, queue, reporter):
    # spinning & polling from queue
    register_reporter(reporter)
    while True:
        args = queue.get()
        if args is None:
            queue.task_done()
            break
        result = run_task(result_table, worker_id, args)
        queue.task_done()


def run():
    n_combinations = 8
    n_episodes = 3000
    # n_runs = 5
    queue = mp.JoinableQueue()
    n_tasks = n_combinations
    n_workers = 4

    start = time.time()

    # queue for mean & std
    result_table = mp.Array('f', n_combinations * n_episodes * 2, lock=False)

    # progress reporter
    reporter = find_reporter()

    for i in range(n_workers):
        p = mp.Process(target=worker, args=(i, result_table, queue, reporter))
        p.start()
        # print('started process', i)

    # send task args to queue
    for i in range(n_combinations):
        queue.put((i, {"xD": 5, "n_episodes": n_episodes}))
    for i in range(n_workers):
        queue.put(None)

    queue.join()
    flush()
    print('ALL DONE', result_table)
    # arr = np.frombuffer(result_table.get_obj())
    # print(np.array(result_table))

    print('time {}s'.format(round(time.time() - start, 4)))


run()
