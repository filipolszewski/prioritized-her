from tools import *


class Evaluator:
    """
    Class for evaluating success rate of given agent on given environment.
    Due to the nature of robotic environments, episodes in which the goal is
    achieved exactly after the reset, are dropped as they add unneeded noise to
    the calculations.

    Evaluator assumes that the agent is using a normalizer for state
    observations.
    """

    @staticmethod
    def test_agent(env, agent):
        test_episodes_count, ep_count, success_count = 50, 0, 0

        while ep_count != test_episodes_count:
            state = agent.normalizer.normalize(env.reset())
            done = False
            cnt = 0
            while not done:
                action = agent.actor_target(
                    flatten_state_dict_for_model(state)).detach().numpy()
                state, reward, done, info = env.step(action)

                if info['is_success'] == 1.0:
                    if cnt == 0:
                        break
                    ep_count += 1
                    success_count += 1
                    break
                # normalize the state
                state = agent.normalizer.normalize(state)
                cnt += 1
            ep_count += 1

        return success_count / test_episodes_count
