from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm, trange
from tqdm.auto import trange
from tqdm import tqdm
from time import sleep
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor


def work(idx):
    progress = tqdm(total=300, desc="process #{}".format(idx))
    for i in range(300):
        sleep(1)
        progress.update(1)


def main():
    ps = []

    for i in range(4):
        p = mp.Process(target=work, args=(i,))
        ps.append(p)
        p.start()

    for p in range(4):
        ps[i].join()


# if __name__ == "__main__":
#     mp.freeze_support()
#     with ProcessPoolExecutor() as pool:
#         main()
# for i in trange(4, desc='1st loop'):
#     for j in trange(5, desc='2nd loop'):
#         for k in trange(50, desc='3rd loop', leave=False):
#             sleep(0.01)


L = list(range(9))


def progresser(n):
    interval = 0.001 / (n + 2)
    total = 5000
    text = "#{}, est. {:<04.2}s".format(n, interval * total)
    for _ in trange(total, desc=text, position=n):
        sleep(interval)


if __name__ == '__main__':
    mp.freeze_support()  # for Windows support
    p = mp.Pool(initializer=tqdm.set_lock, initargs=(mp.Lock(),))
    p.map(progresser, L)
