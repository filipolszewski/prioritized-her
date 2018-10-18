import torch
from tools import *


class Normalizer:
    def __init__(self, obs_len, goal_len):
        num_inputs = obs_len + 2 * goal_len
        self.n = torch.zeros(num_inputs)
        self.mean = torch.zeros(num_inputs)
        self.mean_diff = torch.zeros(num_inputs)
        self.var = torch.zeros(num_inputs)
        self.obs_len = obs_len
        self.goal_len = goal_len

    def observe(self, x):
        x = flatten_state_dict(x)
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = torch.clamp(self.mean_diff / self.n, min=1e-2)

    def normalize(self, state):
        """
        Args:
            state(dict): env state
        """
        obs_std = torch.sqrt(self.var)
        flat = flatten_state_dict(state)
        flat = (flat - self.mean) / obs_std
        return flat_state_to_dict(flat, self.obs_len, self.goal_len)
