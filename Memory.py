import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Sampling should not execute when the tree is not full !!!
class SumTree(object):
    data_pointer = 0

    def __init__(self, capacity, permanent_data=0):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # stores not probabilities but priorities !!!
        self.data = np.zeros(capacity, dtype=object)  # stores transitions
        self.permanent_data = permanent_data  # numbers of data which never be replaced, for demo data protection
        assert 0 <= self.permanent_data <= self.capacity  # equal is also illegal
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.data_pointer

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.full = True
            self.data_pointer = self.data_pointer % self.capacity + self.permanent_data  # make sure demo data permanent

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            if v <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                v -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]


class Memory(object):

    epsilon = 0.001  # small amount to avoid zero priority
    demo_epsilon = 1.0  # 1.0  # extra
    alpha = 0.4  # [0~1] convert the importance of TD error to priority
    beta = 0.6  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity, permanent_data=0):
        self.permanent_data = permanent_data
        self.tree = SumTree(capacity, permanent_data)

    def __len__(self):
        return len(self.tree)

    def full(self):
        return self.tree.full

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max_p for new transition

    def sample(self, n):
        assert self.full()
        b_idx = np.empty((n,), dtype=np.int32)
        b_memory = np.empty((n, self.tree.data[0].size), dtype=object)
        ISWeights = np.empty((n, 1))
        pri_seg = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p
        assert min_prob > 0

        for i in range(n):
            v = np.random.uniform(pri_seg * i, pri_seg * (i + 1))
            idx, p, data = self.tree.get_leaf(v)  # note: idx is the index in self.tree.tree
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i] = idx, data
        return b_idx, b_memory, ISWeights  # note: b_idx stores indexes in self.tree.tree, not in self.tree.data !!!

    # update priority
    def batch_update(self, tree_idxes, abs_errors):
        abs_errors[self.tree.permanent_data:] += self.epsilon
        # priorities of demo transitions are given a bonus of demo_epsilon, to boost the frequency that they are sampled
        abs_errors[:self.tree.permanent_data] += self.demo_epsilon
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idxes, ps):
            self.tree.update(ti, p)





