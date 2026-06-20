import numpy as np
import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position +1 ) % self.capacity

    def sample(self, batch_size):
        state_batch, goal_batch, next_state_batch, ex_reward_batch, done_mask = zip(*random.sample(self.memory, batch_size))
        state_batch = np.array(state_batch)
        goal_batch = np.array(goal_batch)
        next_state_batch = np.array(next_state_batch)
        ex_reward_batch = np.array(ex_reward_batch)
        done_mask = np.array(done_mask)
        return state_batch, goal_batch, next_state_batch, ex_reward_batch, done_mask

    def __len__(self):
        return len(self.memory)
