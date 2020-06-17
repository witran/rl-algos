import multiprocessing as mp
import threading
import time
import random
import numpy as np
import socket
import sys
import concurrent.futures
import pickle


class QueueMonitor():
    def __init__(self, n_workers):
        self._progress_table = [0 for _ in range(n_workers)]
        self._max_progress_table = [1 for _ in range(n_workers)]
        self._n_workers = n_workers
        self._thread = threading.Thread(target=self.loop)
        self._queue = mp.Queue()

    def start(self):
        self._thread.start()

    def loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = ('localhost', 4321)
        while True:
            worker_id, progress, max_progress = self._queue.get()
            if worker_id is None and progress is None and max_progress is None:
                return
            self._progress_table[worker_id] = progress
            self._max_progress_table[worker_id] = max_progress

            # shared render logic
            data = pickle.dumps([list(self._progress_table),
                                 list(self._max_progress_table)])
            # sent = sock.sendto(data, server_address)

    def update(self, worker_id, task_id, progress, max_progress):
        self._queue.put((worker_id, progress, max_progress))

    def stop(self):
        self._queue.put((None, None, None))


class SharedMemMonitor():
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

            # shared render logic
            data = pickle.dumps([list(self._progress_table),
                                 list(self._max_progress_table)])
            # sent = sock.sendto(data, server_address)

    # worker processes will call this fn
    def update(self, worker_id, task_id, progress, max_progress):
        self._progress_table[worker_id] = progress
        self._max_progress_table[worker_id] = max_progress

    def stop(self):
        self._running = False


def hello(anything):
    print('helloooo', mp.current_process().name)


def run_task(worker_id, result_table, monitor, args):
    task_id, params = args
    n_iters = params["n_iters"]
    monitor.update(worker_id, task_id, 0, n_iters)
    for i in range(n_iters):
        time.sleep(0.001)
        monitor.update(worker_id, task_id, i, n_iters)


def worker(worker_id, result_table, monitor, task_queue):
    # spinning & polling from task_queue
    while True:
        args = task_queue.get()
        if args is None:
            task_queue.task_done()
            break
        result = run_task(worker_id, result_table, monitor, args)
        task_queue.task_done()


N_PROCS = 8
N_ITERS = 1 << 12
N_TASKS = 8

mp.set_start_method('fork', force=True)


def test(monitor):
    task_queue = mp.JoinableQueue()
    monitor.start()
    result_table = None

    for i in range(N_PROCS):
        p = mp.Process(target=worker, args=(
            i, result_table, monitor, task_queue))
        p.start()
    # send task args to task_queue
    for i in range(N_TASKS):
        task_queue.put((i, {"n_iters": N_ITERS}))
    for i in range(N_PROCS):
        task_queue.put(None)

    task_queue.join()
    monitor.stop()
    print('done')

    # with concurrent.futures.ProcessPoolExecutor(max_workers=N_PROCS) as executor:
    #     args = [(monitor i, {'n_iters': N_ITERS}) for i in range(N_TASKS)]
    #     for _ in executor.map(run_task, args):
    #         print('hello')
    #     # for _ in executor.map(hello, [None for __ in range(100)]):
    #     #     print('hello')
    #     print('done')
    #     monitor.stop()


if sys.argv[1] == "queue":
    queue_monitor = QueueMonitor(n_workers=N_PROCS)
    start = time.time()
    test(queue_monitor)
    print('queue monitor: {}s'.format(round(time.time() - start, 4)))

else:

    shm_monitor = SharedMemMonitor(n_workers=N_PROCS)
    start = time.time()
    test(shm_monitor)
    print('shm monitor: {}s'.format(round(time.time() - start, 4)))
