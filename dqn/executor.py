import threading
import multiprocessing as mp
import socket
import time
import pickle
import random
import numpy as np

mp.set_start_method('fork', force=True)

# assumptions:
# results returned by fn is uniform
# each args has structure: (task_id, params)
# result will be written into result_table


def launch(fn, args, output_size=1, n_workers=8):
    result_table = mp.Array('f', [0] * len(args) * output_size, lock=False)
    monitor = Monitor(n_workers, len(args))
    queue = mp.JoinableQueue()
    procs = []

    for worker_id in range(n_workers):
        p = mp.Process(target=worker_loop, args=(
            worker_id, queue, result_table, monitor, fn))
        # worker_id, queue, result_table, fn))
        p.start()
        procs.append(p)

    for arg in args:
        queue.put(arg)

    for _ in range(n_workers):
        queue.put(None)

    queue.join()
    for p in procs:
        p.join()
    monitor.join()

    return np.array(result_table)


def worker_loop(worker_id, queue, result_table, monitor, fn):
    while True:
        args = queue.get()
        if args is None:
            queue.task_done()
            break

        task_id = args[0]

        # run function
        result = fn(worker_id, monitor, args)

        for i in range(len(result)):
            result_table[len(result) * task_id + i] = result[i]

        queue.task_done()
        monitor.task_done(worker_id)

    monitor.done(worker_id)


class Monitor():
    def __init__(self, n_workers, n_tasks):
        self._progress_table = mp.Array('i', [0] * n_workers, lock=False)
        self._max_progress_table = mp.Array('i', [1] * n_workers, lock=False)
        self._done = mp.Array('i', [0] * n_workers, lock=False)
        self._task_count_table = mp.Array('i', [0] * n_workers, lock=False)
        self._n_workers = n_workers
        self._thread = threading.Thread(target=self._loop)
        self._thread.start()
        self._total_task_count = n_tasks

    def _all_worker_done(self):
        return all(done == 1 for done in self._done)

    # monitoring thread calls this fn
    def _loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = ('localhost', 4321)
        while not self._all_worker_done():
            time.sleep(0.05 + random.random() * 0.1)
            # time.sleep(0.01)

            task_done_count = sum(self._task_count_table)

            # shared render logic
            data = pickle.dumps([list(self._progress_table),
                                 list(self._max_progress_table),
                                 task_done_count,
                                 self._total_task_count])
            sent = sock.sendto(data, server_address)

    # called by worker processes
    def done(self, worker_id):
        print('worker calls done', worker_id)
        self._progress_table[worker_id] = self._max_progress_table[worker_id]
        self._done[worker_id] = 1

    def task_done(self, worker_id):
        self._task_count_table[worker_id] += 1

    # called by worker processes
    def update(self, worker_id, progress, max_progress):
        self._progress_table[worker_id] = progress
        self._max_progress_table[worker_id] = max_progress

    def join(self):
        self._thread.join()
