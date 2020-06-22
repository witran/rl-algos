import socket
import sys
import pickle
import os

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('localhost', 4321)
sock.bind(server_address)
os.system('clear')

while True:
    data, address = sock.recvfrom(4096)
    progress, max_progress, task_done_count, total_task_count = pickle.loads(
        data)
    n = len(progress)

    code = '\033[1G' + '\033[A' * (n + 1)
    # '\033[1G' move the cursor to the beginning of the line
    # '\033[A' move the cursor up
    # '\033[0J' clear from cursor to end of screen

    sys.stdout.write(code)
    sys.stdout.flush()
    percentages = [(progress[i] / max_progress[i])
                   for i in range(n)]

    n_segments = 40
    bars = [round(percentages[i] * 40) for i in range(n)]

    s = "\n".join(
        'worker #{}: {}% ({}/{}) {}'.format(
            i,
            str(round(percentages[i] * 100, 2)).ljust(6),
            str(progress[i]).ljust(len(str(max_progress[i]))),
            max_progress[i],
            ':' * bars[i] + '-' * (n_segments - bars[i]))
        for i in range(n))
    s = "{}/{} tasks completed\n".format(task_done_count, total_task_count) + s
    sys.stdout.write(s)
    sys.stdout.write('\n')
    sys.stdout.flush()
