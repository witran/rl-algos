import time
import sys
sys.stdout.write('*' * 100 + '\r')
for x in range(100):
    time.sleep(0.02)
    # print('{}\r'.format('.'), end="")
    sys.stdout.write('.' * x + '\r')
