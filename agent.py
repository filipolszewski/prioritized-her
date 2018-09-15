import json
# import sys
# from shutil import copyfile
import torch
import torch.nn.functional as F


class Net(torch.nn.Module):

    def __init__(self, input_size, hidden1_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(input_size, hidden1_size)
        self.fc2 = torch.nn.Linear(hidden1_size, output_size)
        self.output_size = output_size

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class Agent:

    def __init__(self, env):
        with open('./configuration.json') as config_file:
            self.config = json.load(config_file)['agent']

        self.env = env
        pass

    def __str__(self):
        return 'RL_Agent Object'

    def reset(self):
        pass

    def run(self):
        pass


    def _train(self):
        pass


    def get_next_action_greedy(self, state):
        pass

    def _get_next_action_epsilon_greedy(self, state):
        pass

    def get_experience_batch(self):
        pass

    def append_sample_to_memory(self, current_state, action_index, reward,
                                next_state, terminal_state):
        pass

    def huber_loss(self, values, target, weights):
        loss = (torch.abs(values - target) < 1).float() *\
               0.5 * (values - target) ** 2 + \
               (torch.abs(values - target) >= 1).float() *\
               (torch.abs(values - target) - 0.5)

        return (weights * loss).sum()

#
# class AgentUtils:
#     """Abstract class providing save and load methods for Agent objects"""
#
#     @staticmethod
#     def load(agent, model_id):
#         """Loads network configuration and model
#
#         Loads from file into the Agent's
#         network fields.
#
#         Args:
#             agent(Agent): an Agent object, to whom we want to load
#             model_id(int): id of model which we want to load
#
#         """
#         add_path = ''
#         if 'tests' in os.getcwd():
#             add_path = '../'
#         conf_path = add_path + \
#             'saved_models/model_{}/configuration.json'.format(model_id)
#         model_path = add_path + \
#             'saved_models/model_{}/network.pt'.format(model_id)
#
#         # loading configuration file
#         try:
#             with open(conf_path) as config_file:
#                 agent.config = json.load(config_file)['agent']
#             agent.reset()
#         except FileNotFoundError as exc:
#             print("Loading model failed. No model with given index, or no" +
#                   " configuration file. Error: \n")
#             print(exc)
#             sys.exit()
#
#         # load network model
#         try:
#             agent.q_network.load_state_dict(torch.load(model_path))
#             agent.target_network.load_state_dict(torch.load(model_path))
#         except (RuntimeError, AssertionError) as exc:
#             print('Error while loading model. Wrong network size, or not' +
#                   ' an Agent? Aborting. Error:')
#             print(exc)
#             sys.exit()
#
#     @staticmethod
#     def save(model, rewards=None, old_id=None):
#         """Save model, configuration file and training rewards
#
#         Saving to files in the saved_models/{old_id} directory.
#
#         Args:
#             old_id(number): id of the model if it  was loaded, None otherwise
#             model(torch.nn.Net): neural network torch model (q_network)
#             rewards(list): list of total rewards for each episode, default None
#
#         """
#         add_path = ''
#         if 'tests' in os.getcwd():
#             add_path = '../'
#         path = add_path + 'saved_models/model_'
#
#         # create new directory with incremented id
#         new_id = 0
#         while True:
#             if not os.path.exists(path + '{}'.format(new_id)):
#                 os.makedirs(path + '{}'.format(new_id))
#                 break
#             new_id += 1
#
#         # copy old rewards log to append new if model was loaded
#         if old_id:
#             try:
#                 copyfile(
#                     path + '{}/rewards.log'.format(old_id),
#                     path + '{}/rewards.log'.format(new_id))
#             except FileNotFoundError:
#                 print('Warning: no rewards to copy found,\
#                       but OLD ID is not None.')
#
#         #  --- save new data
#         # model
#         torch.save(model.q_network.state_dict(),
#                    path + '{}/network.pt'.format(new_id))
#
#         # config
#         config_path = add_path + '../configuration.json'
#         if old_id:
#             config_path = path + "{}/configuration.json".format(old_id)
#
#         copyfile(config_path, path + "{}/configuration.json".format(new_id))
#
#         if not rewards:
#             return
#         # rewards log
#         with open(path + "{}/rewards.log".format(new_id), "a") as logfile:
#             for reward in rewards:
#                 logfile.write("{}\n".format(reward))
#         # rewards chart
#         rewards = []
#         for line in open(path + '{}/rewards.log'.format(new_id), 'r'):
#             values = [float(s) for s in line.split()]
#             rewards.append(values)
#         avg_rewards = []
#         for i in range(len(rewards) // (10 or 1)):
#             avg_rewards.append(np.mean(rewards[10 * i: 10 * (i + 1)]))
#         plt.plot(avg_rewards)
#         plt.savefig(path + '{}/learning_plot.png'.format(new_id))
#         plt.close()
#
#         return new_id


# class SumTree:
#
#     write = 0
#
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.tree = numpy.zeros(2 * capacity - 1)
#         self.data = numpy.zeros(capacity, dtype=object)
#         self.n_entries = 0
#
#     # update to the root node
#     def _propagate(self, idx, change):
#         parent = (idx - 1) // 2
#         self.tree[parent] += change
#         if parent != 0:
#             self._propagate(parent, change)
#
#     # find sample on leaf node
#     def _retrieve(self, idx, s):
#         left = 2 * idx + 1
#         right = left + 1
#
#         if left >= len(self.tree):
#             return idx
#
#         if s <= self.tree[left]:
#             return self._retrieve(left, s)
#         else:
#             return self._retrieve(right, s - self.tree[left])
#
#     def total(self):
#         return self.tree[0]
#
#     # store priority and sample
#     def add(self, p, data):
#         idx = self.write + self.capacity - 1
#
#         self.data[self.write] = data
#         self.update(idx, p)
#
#         self.write += 1
#         if self.write >= self.capacity:
#             self.write = 0
#
#         if self.n_entries < self.capacity:
#             self.n_entries += 1
#
#     # update priority
#     def update(self, idx, p):
#         change = p - self.tree[idx]
#
#         self.tree[idx] = p
#         self._propagate(idx, change)
#
#     # get priority and sample
#     def get(self, s):
#         idx = self._retrieve(0, s)
#         data_idx = idx - self.capacity + 1
#
#         return idx, self.tree[idx], self.data[data_idx]
#
#
# class Memory:
#     """Based on a SumTree, storing transitions batches for Agent."""
#
#     def __init__(self, max_size, alpha, epsilon, beta, beta_increment):
#         self.sum_tree = SumTree(max_size)
#         self.alpha = alpha
#         self.epsilon = epsilon
#         self.beta = beta
#         self.beta_increment = beta_increment
#
#     def add(self, transition, error):
#         priority = self.get_priority(error)
#         self.sum_tree.add(priority, transition)
#
#     def get_priority(self, error):
#         return (abs(error) + self.epsilon) ** self.alpha
#
#     def update(self, index, error):
#         priority = self.get_priority(error)
#         self.sum_tree.update(index, priority)
#
#     def sample(self, batch_size):
#         batch = []
#         indexes = []
#         priorities = []
#         priority_segment = self.sum_tree.total() / batch_size
#
#         self.beta = np.min([1., self.beta + self.beta_increment])
#
#         for i in range(batch_size):
#             a = i * priority_segment
#             b = (i + 1) * priority_segment
#             value = random.uniform(a, b)
#
#             (index, priority, data) = self.sum_tree.get(value)
#             batch.append(data)
#             indexes.append(index)
#             priorities.append(priority)
#
#         probabilities = priorities / self.sum_tree.total()
#         importance_sampling_weights = np.power(self.sum_tree.n_entries *
#                                                probabilities, -self.beta)
#         importance_sampling_weights /= importance_sampling_weights.max()
#
#         return batch, indexes, Variable(torch.Tensor(
#             importance_sampling_weights))
#
#     def len(self):
#         return self.sum_tree.n_entries
