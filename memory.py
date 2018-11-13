import numpy as np
import torch
import random


class ReplayBuffer(object):
    def __init__(self, size):
        self.storage = []
        self.counter = 0
        self.size = size

    # Expects tuples of (state, reward, action, next_state, done)
    def append(self, data):
        if self.counter < self.size:
            self.storage.append(data)
        else:
            self.storage[self.counter % self.size] = data
        self.counter += 1

    def extend(self, data):
        for x in data:
            self.append(x)

    def get_random_batch(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, r, u, y, d = [], [], [], [], []

        for i in ind:
            X, R, U, Y, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(r).reshape(-1, 1), np.array(u), \
            np.array(y), np.array(d).reshape(-1, 1)

    def __len__(self):
        return len(self.storage)


class SumTree:

    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedMemory:
    """Based on a SumTree, storing transitions batches for Agent."""

    def __init__(self, max_size, alpha, epsilon, beta, beta_increment):
        self.sum_tree = SumTree(max_size)
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta
        self.beta_increment = beta_increment

    def add(self, transition, error):
        priority = self.get_priority(error)
        self.sum_tree.add(priority, transition)

    def get_priority(self, error):
        return (abs(error) + self.epsilon) ** self.alpha

    def update(self, index, error):
        priority = self.get_priority(error)
        self.sum_tree.update(index, priority)

    def sample(self, batch_size):
        batch = []
        indexes = []
        priorities = []
        priority_segment = self.sum_tree.total() / batch_size

        self.beta = np.min([1., self.beta + self.beta_increment])

        for i in range(batch_size):
            a = i * priority_segment
            b = (i + 1) * priority_segment
            data = 0
            while data == 0:
                # UGLY hack until I find reason why sometimes the tree
                # returns 0 instead of tuple
                value = random.uniform(a, b)
                (index, priority, data) = self.sum_tree.get(value)
                batch.append(data)
                indexes.append(index)
                priorities.append(priority)

        probabilities = priorities / self.sum_tree.total()
        probabilities[probabilities == 0] = 1e-10
        importance_sampling_weights = np.power(self.sum_tree.n_entries *
                                               probabilities, -self.beta)
        importance_sampling_weights /= importance_sampling_weights.max()

        return batch, indexes, torch.Tensor(
            importance_sampling_weights)

    def __len__(self):
        return self.sum_tree.n_entries
