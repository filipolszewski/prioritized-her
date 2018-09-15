import gym
import json
from gym import logger
from agent import Agent


def get_config_dict():
    with open('./configuration.json') as config_file:
        return json.load(config_file)['main']


def plot_rewards():
    pass


if __name__ == '__main__':

    logger.set_level(logger.INFO)

    config = get_config_dict()

    env = gym.make('HandManipulateBlock-v0')
    env.seed(config['seed'])
    agent = Agent(env)

    rewards = []

    for i in range(config['training_episodes']):
        total_reward = agent.run()
        rewards.append(total_reward)
