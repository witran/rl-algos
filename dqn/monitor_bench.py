import multiprocessing as mp
import threading
import time
import random
import numpy as np
import socket
import sys
import concurrent.futures
import pickle
import cProfile


N_PROCS = 8
N_ITERS = 1 << 10
N_TASKS = 16
TASK_DURATION = 0.0001
N_TESTS = 5


def run_test():
    if sys.argv[1] == "queue":
        s = 0
        for i in range(N_TESTS):
            queue_monitor = QueueMonitor(n_workers=N_PROCS)
            queue_monitor.start()
            start = time.time()
            test(queue_monitor)
            d = time.time() - start
            print('run #{}: {}s'.format(i, round(d, 4)))
            s += time.time() - start
            queue_monitor.stop()

        mean = s / N_TESTS
        print('queue mean: {}s'.format(round(mean, 4)))
        print('span duration: {}s'.format(
            round(N_TASKS / N_PROCS * N_ITERS * TASK_DURATION, 4)))

    else:
        s = 0
        for i in range(N_TESTS):
            shm_monitor = SharedMemMonitor(n_workers=N_PROCS)
            shm_monitor.start()
            start = time.time()
            test(shm_monitor)
            d = time.time() - start
            print('run #{}: {}s'.format(i, round(d, 4)))
            s += d
            shm_monitor.stop()

        mean = s / N_TESTS
        print('shm mean: {}s'.format(round(mean, 4)))
        print('span duration: {}s'.format(
            round(N_TASKS / N_PROCS * N_ITERS * TASK_DURATION, 4)))


def test(monitor):
    task_queue = mp.JoinableQueue()
    result_table = None
    ps = []

    for i in range(N_PROCS):
        p = mp.Process(target=worker_profiled, args=(
            i, result_table, monitor, task_queue))
        ps.append(p)
        p.start()
    # send task args to task_queue
    for i in range(N_TASKS):
        task_queue.put((i, {"n_iters": N_ITERS}))
    for i in range(N_PROCS):
        task_queue.put(None)

    task_queue.join()
    # monitor.stop()

    # stuck here because resource i never freed because None is sent too early and queue is blocking
    # somehow writer don't exit because reader hasn't consumed???
    print('task_queue joined, monitor stopped')
    for i in range(N_PROCS):
        ps[i].join()
    print('workers joined')

    # with concurrent.futures.ProcessPoolExecutor(max_workers=N_PROCS) as executor:
    #     args = [(monitor i, {'n_iters': N_ITERS}) for i in range(N_TASKS)]
    #     for _ in executor.map(run_task, args):
    #         print('hello')
    #     # for _ in executor.map(hello, [None for __ in range(100)]):
    #     #     print('hello')
    #     print('done')
    #     monitor.stop()


class QueueMonitor():
    def __init__(self, n_workers):
        self._progress_table = [0] * n_workers
        self._max_progress_table = [1] * n_workers
        self._n_workers = n_workers
        self._thread = threading.Thread(target=self.loop)
        self._queue = mp.Queue()

        self.put_count = [0] * n_workers
        self.get_count = [0] * n_workers
        self.done_count = 0

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._thread.start()

    def loop(self):
        server_address = ('localhost', 4321)
        while True:
            args = self._queue.get()
            if args is None:
                self._sock.close()
                return

            worker_id, progress, max_progress = args
            if progress is None:
                self.done_count += 1
                if self.done_count == self._n_workers:
                    self._sock.close()
                    return
                continue

            self._progress_table[worker_id] = progress
            self._max_progress_table[worker_id] = max_progress

            self.get_count[worker_id] += 1
            # print('get count', self.get_count[worker_id], worker_id)

            # shared render logic
            data = pickle.dumps([list(self._progress_table),
                                 list(self._max_progress_table)])
            sent = self._sock.sendto(data, server_address)

    def update(self, worker_id, task_id, progress, max_progress):
        self.put_count[worker_id] += 1
        # print('put count', self.put_count[worker_id], worker_id)
        self._queue.put((worker_id, progress, max_progress))

    def done(self, worker_id):
        self._queue.put((worker_id, None, None))

    def stop(self):
        # print('stop')
        # self._queue.put(None)
        # self._sock.close()
        # print('here')
        self._thread.join()
        # print('finish thread join')


class SharedMemMonitor():
    def __init__(self, n_workers):
        self._progress_table = mp.Array('i', [0] * n_workers, lock=False)
        self._max_progress_table = mp.Array('i', [1] * n_workers, lock=False)
        self._running = True
        self._n_workers = n_workers
        self._thread = threading.Thread(target=self.loop)
        self._done = mp.Array('i', [0] * n_workers, lock=False)

    def start(self):
        self._thread.start()

    def _is_done(self):
        return all(done == 1 for done in self._done)

    # monitoring thread calls this fn

    def loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = ('localhost', 4321)
        # while self._running:
        # while not self._is_done():
        while self._running:
            # time.sleep(0.05 + random.random() * 0.1)
            time.sleep(0.01)

            # shared render logic
            data = pickle.dumps([list(self._progress_table),
                                 list(self._max_progress_table)])
            sent = sock.sendto(data, server_address)

        # just make everyone 100%
        # or have a report table for error status
        for i in range(self._n_workers):
            self._progress_table[i] = self._max_progress_table[i]
        # shared render logic
        data = pickle.dumps([list(self._progress_table),
                             list(self._max_progress_table)])
        sent = sock.sendto(data, server_address)

    # worker processes will call this fn
    def update(self, worker_id, task_id, progress, max_progress):
        self._progress_table[worker_id] = progress
        self._max_progress_table[worker_id] = max_progress

    # worker processes will call this when worker done
    def done(self, worker_id):
        return  # no op
        self._progress_table[worker_id] = self._max_progress_table[worker_id]
        self._done[worker_id] = 1

    def stop(self):
        self._running = False
        self._thread.join()


def run_task(worker_id, result_table, monitor, args):
    task_id, params = args
    n_iters = params["n_iters"]
    monitor.update(worker_id, task_id, 0, n_iters)
    for i in range(n_iters):
        time.sleep(TASK_DURATION)
        monitor.update(worker_id, task_id, i + 1, n_iters)


def worker(worker_id, result_table, monitor, task_queue):
    while True:
        args = task_queue.get()
        if args is None:
            task_queue.task_done()
            monitor.done(worker_id)
            break
        result = run_task(worker_id, result_table, monitor, args)
        task_queue.task_done()
    # print('task done')


def worker_profiled(worker_id, result_table, monitor, task_queue):
    # if worker_id == 0:
    #     cProfile.runctx('worker(worker_id, result_table, monitor, task_queue)',
    #                     globals(), locals(), 'monitor_{}_{}.cprofile'.format(
    #                         sys.argv[1], worker_id))
    # else:
    #     worker(worker_id, result_table, monitor, task_queue)
    worker(worker_id, result_table, monitor, task_queue)


mp.set_start_method('fork', force=True)


run_test()
