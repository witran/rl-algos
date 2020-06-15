import numpy as np
import time


def prefix_sum(a):
    p_sum = [a[0]]
    for i in range(1, len(a)):
        p_sum.append(a[i] + p_sum[i - 1])

    return p_sum


def binary_search(x, a):
    # a represents the range: [a[i - 1], a[i]]
    # return index i such that a[i - 1] <= x <= a[i]
    low = 0
    high = len(a)
    while low < high:
        mid = (low + high) // 2
        if x > a[mid]:
            low = mid + 1
        elif x < a[mid]:
            high = mid
        else:
            return mid
    return high


def test_binary_search():
    a = [1, 2, 5, 7, 8, 11, 15]
    print(binary_search(9, a))
    print(binary_search(8, a))
    print(binary_search(1, a))
    print(binary_search(11, a))
    print(binary_search(14, a))
    print(binary_search(4, a))


# test_binary_search()


def choice(a, size, priority=None):
    result = []
    p_sum = prefix_sum(priority)

    for _ in range(size):
        v = np.random.rand() * p_sum[len(p_sum) - 1]
        idx = binary_search(v, p_sum)
        result.append(a[idx])

    return result


def pretty(float_arr):
    return list(map(lambda x: round(x, 4), float_arr))


def test_choice():
    n = 1 << 5
    size = 1 << 3
    a = [i for i in range(n)]
    n_test = 100000
    count = [0 for _ in range(n)]

    time.date()
    priority = [np.random.randint(0, 100) for _ in range(n)]

    # print(a)
    # print(choice(a, size, priority))
    # return

    for i in range(n_test):
        for v in choice(a, size, priority=priority):
            count[v] += 1

    total_count = sum(count)
    observed_probs = [count[i] / total_count for i in range(n)]
    total_priority = sum(priority)
    given_probs = [priority[i] / total_priority for i in range(n)]
    print('observed'.ljust(10), pretty(observed_probs))
    print('given'.ljust(10), pretty(given_probs))


test_choice()
