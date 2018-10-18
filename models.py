import torch.nn.functional as F
import numpy as np
import torch


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class ActorNet(torch.nn.Module):
    def __init__(self, input_size, h1_size, h2_size, h3_size, output_size,
                 max_action):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, h1_size)
        self.fc2 = torch.nn.Linear(h1_size, h2_size)
        self.fc3 = torch.nn.Linear(h2_size, h3_size)
        self.fc4 = torch.nn.Linear(h3_size, output_size)
        self.max_action = max_action
        self.action_preact = None
        self.init_weights()

    def init_weights(self):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data = torch.Tensor(self.fc4.weight.data.size()) \
            .uniform_(-0.003, 0.003)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        self.action_preact = torch.tanh(x) * self.max_action
        return torch.tanh(x) * self.max_action


class CriticNet(torch.nn.Module):
    def __init__(self, input_size, h1_size, h2_size, h3_size, actions_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, h1_size)
        self.fc2 = torch.nn.Linear(h1_size + actions_size, h2_size)
        self.fc3 = torch.nn.Linear(h2_size, h3_size)
        self.fc4 = torch.nn.Linear(h3_size, 1)
        self.init_weights()

    def init_weights(self):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())
        self.fc4.weight.data = torch.Tensor(self.fc4.weight.data.size()) \
            .uniform_(-0.003, 0.003)

    def forward(self, x, a):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(torch.cat([x, a], 1))
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        return self.fc4(x)