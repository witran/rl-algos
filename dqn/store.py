import numpy as np
import torch

"""
Store
    capacity
        is store size

    items
        is a double-ended queue containing tuple (data, priority)

    tail
        is queue's tail

    sum_tree
        is a sum segment tree on top of the queue
        used to randomly sample according to distribution in log(n)

    min_tree
        is a min segment tree on top of the queue
        used to compute max IS weight to scale down sampled IS weight

implementation notes
    starting from i = 1, node[i] will be parent of node[i * 2], node[i * 2 + 1]
    variable name "index" will be used to index self.items
    variable name "node" will be used to index self.sum_tree & self.min_tree
"""


def to_numpy(samples):
    indexes, s, a, r, s_next, done = [], [], [], [], [], []
    for index, data in samples:
        indexes.append(samples)
        s.append(data[0])
        a.append(data[1])
        r.append(data[2])
        s_next.append(data[3])
        done.append(data[4])

    return indexes, tuple(map(np.array), (s, a, r, s_next, done))


def to_torch(samples):
    pass


class Store:
    def __init__(self, capacity=1 << 20, alpha=1):
        self.capacity = capacity
        self.alpha = alpha
        self.items = [None for _ in range(capacity)]
        self.sum_tree = [0 for _ in range(capacity * 2)]
        self.min_tree = [1 << 30 for _ in range(capacity * 2)]
        self.tail = 0

    def _update_tree(self, index):
        node = index + self.capacity
        self.sum_tree[node] = self.items[index][1]
        self.min_tree[node] = self.items[index][1]
        while node > 1:
            node //= 2
            self.sum_tree[node] = sum(
                self.sum_tree[node * 2], self.sum_tree[node * 2 + 1])
            self.min_tree[node] = min(
                self.min_tree[node * 2], self.min_tree[node * 2 + 1])

    def add(self, item):
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            self.items[self.tail] = item

        self._update_tree(self.tail)
        self.tail = (self.tail + 1) % self.capacity

    def _search(self, priority):
        node = 1
        p = priority
        while node < self.capacity * 2:
            if p > self.sum_tree[node * 2]:
                p -= node.sum_tree[node * 2]
                node = node * 2 + 1
            else:
                node = node * 2
        index = node - self.capacity
        return index, self.items[index][0]

    def _get_importance_weight(self, priorities, beta):
        weights = []
        max_weight = (1 / (self.min_tree[1] * len(self.items))) ** beta
        for p in priorities:
            weights.append(1 / (p * len(self.items)) ** beta)
        return weights

    def sample(self, size=32, beta, format="numpy"):
        priorities = []
        samples = []
        p_sum = self.sum_tree[0]
        for _ in size:
            p_sample = np.random.rand() * p_sum
            samples.append(self._search(p_sample))
            priorities.append(p_sample)

        indexes, data = to_numpy(samples)
        weights = self._get_importance_weight(priorities, beta)

        return indexes, data, weights

    def update_priorities(self, indexes, priorities):
        for i in range(len(indexes)):
            self.items[indexes[i]][1] = priorities[i]
            self._update_tree(indexes[i])


def test():
    store = Store()
    store.add([])
    samples, indexes = store.sample(size)
    store.update_priority(indexes, priorities)
