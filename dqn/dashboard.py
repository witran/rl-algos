import socket
import sys
import pickle

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = ('localhost', 4321)
# print(sys.stderr, 'server started at {}'.format(server_address))
# TODO - how to actually delete
sock.bind(server_address)

while True:
    data, address = sock.recvfrom(4096)
    progress, max_progress, task_done_count, total_task_count = pickle.loads(
        data)
    nlines = len(progress)

    code = '\033[1G' + '\033[A'*(nlines + 1) + '\033[0J'
    # '\033[1G' move the cursor to the beginning of the line
    # '\033[A' move the cursor up
    # '\033[0J' clear from cursor to end of screen

    sys.stdout.write(code)
    s = "\n".join(
        'worker #{}, {}% ({}/{})'.format(
            i, str(round(progress[i] / max_progress[i] * 100, 2)).ljust(6), progress[i], max_progress[i])
        for i in range(len(progress)))
    s = "{}/{} tasks completed\n".format(task_done_count, total_task_count) + s
    sys.stdout.write(s)
    sys.stdout.write('\n')
    sys.stdout.flush()
