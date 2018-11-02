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

        # experimental values
        # self.mean = \
        #     torch.Tensor([1.3060e+00, 7.5912e-01, 4.7468e-01, 1.3419e+00,
        #                   7.4862e-01, 4.2289e-01, 3.5408e-02, -1.0533e-02,
        #                   -5.0285e-02, -4.0495e-08, 3.6686e-05, -8.6649e-04,
        #                   -4.3643e-04, -1.0382e-04, 1.1019e-03, -2.9179e-04,
        #                   -1.4800e-03, -5.3790e-05, -2.5022e-06, -8.6738e-06,
        #                   -1.1097e-03, 2.9676e-04, 1.4604e-03, 9.1511e-04,
        #                   9.3560e-04, 1.3409e+00, 7.5010e-01, 4.2470e-01,
        #                   1.3419e+00, 7.4862e-01, 4.2289e-01])
        #
        # self.var = torch.Tensor(
        #     [0.0153, 0.0213, 0.0100, 0.0106, 0.0106, 0.0100, 0.0309, 0.0353,
        #      0.0100, 0.0100, 0.0100, 0.0210, 0.0100, 0.0160, 0.0100, 0.0100,
        #      0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100, 0.0100,
        #      0.0100, 0.0100, 0.0100, 0.0100, 0.0106, 0.0106, 0.0100])

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
