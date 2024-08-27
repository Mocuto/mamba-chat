import random
import numpy as np
from SumTree import SumTree
# Adapted from source at: https://github.com/rlcode/per


class PrioritizedReplayMemory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity, fn):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.fn = fn
    def __len__(self):
      return self.tree.n_entries

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def push(self, error, *args):
        p = self._get_priority(error)
        if self.fn is not None:
          self.tree.add(p, self.fn(*args))
        else:
          self.tree.add(p, args[0])

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()

        return batch, idxs

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def clear(self):
        self.tree.write = 0
        self.tree.n_entries = 0