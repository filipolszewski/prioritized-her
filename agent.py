import json
import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from shutil import copyfile
import torch
from torch.optim import Adam
import torch.nn.functional as F
import copy
from memory import ReplayBuffer
from noise import OrnsteinUhlenbeckProcess


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
        self.action_preact = x.clone()
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


class Agent:
    def __init__(self, env):
        """Args:
            env(gym.Core.env): environment
        """

        with open('./configuration.json') as config_file:
            self.config = json.load(config_file)['agent']

        self.env = env
        self.state = None
        self.epsilon = None
        self.epsilon_decay = None
        self.state_size = None
        self.actions_size = None
        self.actor = None
        self.actor_target = None
        self.critic = None
        self.critic_target = None
        self.actor_optim = None
        self.critic_optim = None
        self.gamma = None
        self.memory = None
        self.batch_size = None
        self.action_space = None
        self.state_space = None
        self.random_process = None

    def __str__(self):
        return 'RL_Agent Object'

    def reset(self):
        self.action_space = self.env.action_space
        self.state_space = self.env.observation_space

        self.state_size = self.state_space.spaces['observation'].shape[0] + \
            self.state_space.spaces['desired_goal'].shape[0]
        self.actions_size = self.action_space.shape[0]

        max_action = float(self.env.action_space.high[0])

        self.actor = ActorNet(self.state_size, *self.config['net_sizes'],
                              self.actions_size, max_action)
        self.critic = CriticNet(self.state_size, *self.config['net_sizes'],
                                self.actions_size)

        self.actor_target = ActorNet(self.state_size, *self.config['net_sizes'],
                                     self.actions_size, max_action)
        self.critic_target = CriticNet(self.state_size,
                                       *self.config['net_sizes'],
                                       self.actions_size)

        # hard copy
        self.update(self.critic_target, self.critic, 1)
        self.update(self.actor_target, self.actor, 1)

        self.actor_optim = Adam(self.actor.parameters(),
                                lr=self.config['learning_rate'] / 10)
        self.critic_optim = Adam(self.critic.parameters(),
                                 lr=self.config['learning_rate'],
                                 weight_decay=1e-2)

        self.epsilon = self.config['epsilon']
        self.epsilon_decay = self.config['epsilon_decay']
        self.gamma = self.config['gamma']
        self.memory = ReplayBuffer(self.config['memory_size'])
        self.batch_size = self.config['batch_size']

        self.random_process = OrnsteinUhlenbeckProcess(
            size=self.actions_size, theta=self.config['ou_theta'],
            mu=self.config['ou_mu'], sigma=self.config['ou_sigma'])

    def run_presentation(self):
        total_reward = 0
        done = False
        self.state = self.env.reset()

        while not done:
            self.env.render()
            action = self._get_action_greedy(self.state)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            self.state = obs
        return total_reward

    def run(self, train):
        total_reward = 0
        done = False
        self.state = self.env.reset()
        ep_transitions = []
        while not done:
            if self.config['render']:
                self.env.render()
            action = self._get_action_epsilon_greedy(self.state)
            obs, reward, done, info = self.env.step(action)

            total_reward += reward

            transition = [self.state, reward, action, obs, not done]
            ep_transitions.append(transition)
            self.memory.append(copy.deepcopy((
                self.obs_to_state_with_desired_goal(self.state), reward,
                action, self.obs_to_state_with_desired_goal(obs), not done)))
            self.state = obs

        if random.random() < self.config["her-probability"]:
            self._generate_her_transitions(ep_transitions)

        if len(self.memory) > self.batch_size * 5 and train:
            for i in range(10):
                batch = self.memory.get_random_batch(self.config['batch_size'])
                self._train(batch)
            self.update_networks()

        if self.epsilon > self.config['epsilon_min']:
            self.epsilon *= self.epsilon_decay

        return total_reward

    def _generate_her_transitions(self, transitions):
        if self.config['her-type'] == 'future':
            for i, t in enumerate(transitions[:-1]):
                for k in range(self.config['her-k_value']):
                    future_idx = random.randrange(i + 1, 50)
                    future_goal = transitions[future_idx][3]['achieved_goal']
                    her_transition = self._make_her_transition(t, future_goal)
                    self.memory.append(her_transition)

        else:
            final_goal = transitions[-1][3]['achieved_goal']
            for t in transitions:
                self.memory.append(self._make_her_transition(t, final_goal))

    def _make_her_transition(self, t, new_goal):
        t = copy.deepcopy(t)
        t[0]['desired_goal'] = new_goal
        t[3]['desired_goal'] = new_goal
        t[1] = self.env.compute_reward(t[3]['achieved_goal'],
                                       t[3]['desired_goal'], None)
        t[0] = self.obs_to_state_with_desired_goal(t[0])
        t[3] = self.obs_to_state_with_desired_goal(t[3])
        return t

    def _train(self, batch):
        state_batch = torch.Tensor(batch[0])
        reward_batch = torch.Tensor(batch[1])
        action_batch = torch.Tensor(batch[2])
        next_state_batch = torch.Tensor(batch[3])
        mask_batch = torch.Tensor(batch[4] * 1)

        next_q_values = self.critic_target(next_state_batch,
                                           self.actor_target(next_state_batch))
        expected_q_values = reward_batch + \
            (self.gamma * mask_batch * next_q_values).detach().clamp_(-50., 0.)

        self.critic_optim.zero_grad()
        q_values = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(q_values, expected_q_values)

        critic_loss.backward()

        self.critic_optim.step()

        self.actor_optim.zero_grad()
        policy_loss = self.critic(state_batch, self.actor(state_batch))
        action_reg = (self.actor.action_preact**2).mean() * 0.5
        policy_loss = -policy_loss.mean() + action_reg
        policy_loss.backward()

        self.actor_optim.step()

    def _get_action_greedy(self, state):
        return self.actor(
            self.obs_to_state_with_desired_goal(state)).detach().numpy()

    def _get_action_epsilon_greedy(self, state):
        if random.random() > self.epsilon:
            action = self._get_action_greedy(state) + \
                     self.random_process.sample() * 0.5
            return np.clip(action, -1., 1.)
        else:
            return self.env.action_space.sample()

    def get_experience_batch(self):
        batch = self.memory.get_random_batch(self.batch_size)
        exp_batch = [0, 0, 0, 0, 0]
        for i in [0, 1, 2, 3, 4]:
            exp_batch[i] = [x[i] for x in batch]
        return exp_batch

    def obs_to_state_with_desired_goal(self, obs):
        return torch.cat((torch.Tensor(obs['observation']),
                          torch.Tensor(obs['desired_goal'])))

    def update_networks(self):
        self.update(self.critic_target, self.critic,
                    self.config['network_update_amount'])
        self.update(self.actor_target, self.actor,
                    self.config['network_update_amount'])

    def update(self, target, src, amount):
        for target_param, param in zip(target.parameters(),
                                       src.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - amount) + param.data * amount)


class AgentUtils:
    """Class providing save and load methods for Agent objects"""

    @staticmethod
    def load(agent, model_id):
        """Loads network configuration and model

        Loads from file into the Agent's
        network fields.

        Args:
            agent(Agent): an Agent object, to whom we want to load
            model_id(str): id of model which we want to load

        """

        conf_path = 'saved_models/model_{}/configuration.json'.format(model_id)
        model_critic_path = 'saved_models/model_{}/critic_network.pt'.format(
            model_id)
        model_actor_path = 'saved_models/model_{}/actor_network.pt'.format(
            model_id)

        # loading configuration file
        try:
            with open(conf_path) as config_file:
                agent.config = json.load(config_file)['agent']
            agent.reset()
        except FileNotFoundError as exc:
            print("Loading model failed. No model with given index, or no" +
                  " configuration file. Error: \n")
            print(exc)
            sys.exit()

        # load network model
        try:
            agent.critic.load_state_dict(torch.load(model_critic_path))
            agent.critic_target.load_state_dict(torch.load(model_critic_path))
            agent.actor.load_state_dict(torch.load(model_actor_path))
            agent.actor_target.load_state_dict(torch.load(model_actor_path))
        except RuntimeError as exc:
            print('Error while loading model. Wrong network size, or not' +
                  ' an Agent? Aborting. Error:')
            print(exc)
            sys.exit()

    @staticmethod
    def save(model, rewards=None, old_id=None):
        """Save model, configuration file and training rewards

        Saving to files in the saved_models/{old_id} directory.

        Args:
            old_id(number): id of the model if it  was loaded, None otherwise
            model(torch.nn.Net): neural network torch model (q_network)
            rewards(list): list of total rewards for each episode, default None

        """

        path = 'saved_models/model_{}'

        # create new directory with incremented id
        new_id = 0
        while True:
            if not os.path.exists(path.format(new_id)):
                os.makedirs(path.format(new_id))
                break
            new_id += 1

        # copy old rewards log to append new if model was loaded
        if old_id:
            try:
                copyfile(
                    (path + '/rewards.log').format(old_id),
                     (path + '/rewards.log').format(new_id))
            except FileNotFoundError:
                print('Warning: no rewards to copy found,\
                      but OLD ID is not None.')

        # --- save new data
        # model

        torch.save(model.critic.state_dict(),
                   (path + '/critic_network.pt').format(new_id))
        torch.save(model.actor.state_dict(),
                   (path + '/actor_network.pt').format(new_id))

        # config
        config_path = 'configuration.json'
        if old_id:
            config_path = (path + "/configuration.json").format(old_id)

        copyfile(config_path, (path + "/configuration.json").format(new_id))

        if not rewards:
            return

        # rewards log
        with open((path + "/rewards.log").format(new_id), "a") as logfile:
            for reward in rewards:
                logfile.write("{}\n".format(reward))

        # rewards chart
        rewards = []
        for line in open((path + '/rewards.log').format(new_id), 'r'):
            values = [float(s) for s in line.split()]
            rewards.append(values)
        avg_rewards = []
        for i in range(len(rewards) // (10 or 1)):
            avg_rewards.append(np.mean(rewards[10 * i: 10 * (i + 1)]))
        plt.plot(avg_rewards)
        plt.savefig((path + '/learning_plot.png').format(new_id))
        plt.close()

        return new_id
