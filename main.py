import json

import gym
import gym.spaces
import matplotlib.pyplot as plt
import numpy as np

from agent import Agent
from agent import AgentUtils
from evaluator import Evaluator


def main(config):
    env = gym.make(config['env_name'])
    # env.env.reward_type = "dense"
    agent = Agent(env)

    rewards = []
    success_rates = []
    agent.reset()

    for i in range(config['training_episodes']):

        if config['save_periodically']:
            if i > 0 and i % 10000 == 0:
                AgentUtils.save(agent, rewards, success_rates)

        if i > 0 and i % 250 == 0:
            success_rate = Evaluator.test_agent(env, agent)
            print("Success rate after {} episodes: {}".format(i, success_rate))
            success_rates.append(success_rate)

        train = (i % 16 == 0)
        total_reward = agent.run(train)
        rewards.append(total_reward)

        if config['print_stats']:
            print_episode_stats(i, config['training_episodes'], total_reward)

    if config['save_experiment']:
        AgentUtils.save(agent, rewards, success_rates)

    if config['make_total_reward_plot']:
        plot_total_rewards(rewards, config['training_episodes'], avg=100)


def print_episode_stats(episode, limit, reward):
    print("{}/{} episode - Total reward: {}".format(episode, limit, reward))


def get_config_dict():
    with open('./configuration.json') as config_file:
        return json.load(config_file)['main']


def plot_total_rewards(rewards, num_episodes, avg=10):
    avg_rewards = []
    for i in range(num_episodes // (avg or 1)):
        avg_rewards.append(np.mean(rewards[avg * i: avg * (i + 1)]))

    plt.plot(avg_rewards)
    plt.show()


if __name__ == "__main__":
    main(config=get_config_dict())
