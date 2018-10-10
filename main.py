import json

import gym
import gym.spaces
import matplotlib.pyplot as plt
import numpy as np
from gym import logger

from agent import Agent
from agent import AgentUtils


def main(config):
    logger.set_level(logger.INFO)

    env = gym.make(config['env_name'])
    env.env.reward_type = "dense"
    # env.seed(config['seed'])
    agent = Agent(env)

    model_id = None
    if config['load_agent_model']:
        model_id = input('Model ID:\n')
        AgentUtils.load(agent, model_id)

    rewards = []

    agent.reset()

    for i in range(config['training_episodes']):

        if config['save_periodically']:
            if i > 0 and i % 1000 == 0:
                model_id = AgentUtils.save(agent, rewards, model_id)
        total_reward = agent.run()
        rewards.append(total_reward)

        if config['print_stats']:
            print_episode_stats(i, config['training_episodes'], total_reward)

    if config['save_experiment']:
        AgentUtils.save(agent, rewards, model_id)

    if config['make_total_reward_plot']:
        plot_total_rewards(rewards, config['training_episodes'], avg=10)


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
