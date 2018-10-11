import gym
import gym.spaces

from agent import Agent
from agent import AgentUtils

"""Shorter script for live presentation of the given model without learning"""
def main():
    env = gym.make("FetchReach-v1")
    # env.seed(1)
    agent = Agent(env)

    model_id = input('Model ID:\n')
    agent.reset()
    AgentUtils.load(agent, model_id)

    for i in range(1000):
        total_reward = agent.run_presentation()
        print("{}/{} episode - Total reward: {}".format(i, 1000, total_reward))


if __name__ == "__main__":
    main()
