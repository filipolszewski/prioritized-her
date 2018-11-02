import gym
import gym.spaces

from agent import Agent
from agent import AgentUtils

"""Shorter script for live presentation of the given model without learning"""

# TODO: Save and load agent's normalizer values for stability and
# TODO: reproducibility during presentation


def main():
    env = gym.make("FetchPush-v1")
    agent = Agent(env)

    model_id = input('Model ID:\n')
    agent.reset()
    AgentUtils.load(agent, model_id)

    while True:
        _run_presentation(agent, env)


def _run_presentation(agent, env):
    done = False
    state = env.reset()
    agent.normalizer.observe(state)
    state = agent.normalizer.normalize(state)

    while not done:
        env.render()
        action = agent.get_action_greedy(state)
        state, reward, done, info = env.step(action)
        agent.normalizer.observe(state)
        state = agent.normalizer.normalize(state)


if __name__ == "__main__":
    main()
