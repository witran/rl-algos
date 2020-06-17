import multiprocessing as mp
import threading
import time
import random
import numpy as np
import socket
import sys
import pickle

mp.set_start_method('fork', force=True)


def run_task(worker_id, result_table, monitor, args):
    task_id, params = args
    n_episodes = params["n_episodes"]

    monitor.update(worker_id, task_id, 0, n_episodes)
    for i in range(n_episodes):
        id_mean = task_id * n_episodes * 2 + i
        id_std = id_mean + n_episodes
        time.sleep(random.random() * 0.005)
        result_table[id_mean] = random.random()
        result_table[id_std] = random.random()
        monitor.update(worker_id, task_id, i, n_episodes)


def worker(worker_id, result_table, monitor, task_queue):
    # spinning & polling from task_queue
    while True:
        args = task_queue.get()
        if args is None:
            task_queue.task_done()
            break
        result = run_task(worker_id, result_table, monitor, args)
        task_queue.task_done()


class Monitor():
    def __init__(self, n_workers):
        self._progress_table = mp.Array(
            'i', [0 for _ in range(n_workers)], lock=False)
        self._max_progress_table = mp.Array(
            'i', [1 for _ in range(n_workers)], lock=False)
        self._running = True
        self._n_workers = n_workers
        self._thread = threading.Thread(target=self.loop)

    def start(self):
        self._thread.start()

    # monitoring thread calls this fn
    def loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = ('localhost', 4321)
        while self._running:
            time.sleep(0.1)
            sent = sock.sendto(
                pickle.dumps([
                    list(self._progress_table),
                    list(self._max_progress_table)]),
                server_address)

    # worker processes will call this fn
    def update(self, worker_id, task_id, progress, max_progress):
        self._progress_table[worker_id] = progress
        self._max_progress_table[worker_id] = max_progress

    def stop(self):
        self._running = False


def run():
    n_combinations = 8
    n_episodes = 3000
    task_queue = mp.JoinableQueue()
    n_tasks = n_combinations
    n_workers = 4

    start = time.time()

    # shm for mean & std result
    result_table = mp.Array('f', n_combinations * n_episodes * 2, lock=False)

    monitor = Monitor(n_workers)
    monitor.start()

    for i in range(n_workers):
        p = mp.Process(target=worker_profiled, args=(
            i, result_table, monitor, task_queue))
        p.start()

    # send task args to task_queue
    for i in range(n_combinations):
        task_queue.put((i, {"n_episodes": n_episodes}))
    for i in range(n_workers):
        task_queue.put(None)

    task_queue.join()
    monitor.stop()
    # print('ALL DONE', result_table)
    # print(np.array(result_table))
    # print('time {}s'.format(round(time.time() - start, 4)))


run()
