import torch


def flatten_state_dict_for_model(state):
    return torch.cat((torch.Tensor(state['observation']),
                      torch.Tensor(state['desired_goal'])))


def flatten_state_dict(state):
    return torch.cat((torch.Tensor(state['observation']),
                      torch.Tensor(state['desired_goal']),
                      torch.Tensor(state['achieved_goal'])))


def flat_state_to_dict(state, obs_len, goal_len):
    d = dict()
    d['observation'] = state[:obs_len]
    d['desired_goal'] = state[obs_len:-goal_len]
    d['achieved_goal'] = state[-goal_len:]
    return d

