import numpy as np
from tools import *
import copy


def generate_her_transitions(transitions, reward_fn, her_type='future',
                             her_k=8):
    """Function that perform Hindsight Experience Replay - generates HER
    transitions from given list of real transitions. Uses reward_fn argument to
    recalculate the rewards for new transitions with substituted goal.
    """

    her_transitions = []
    final_goal = transitions[-1][3]['achieved_goal']
    for idx, t in enumerate(transitions):
        if her_type == 'final':
            her_transition = make_her_transition(t, final_goal, reward_fn)
            her_transitions.append(her_transition)
            continue

        for k in range(her_k):
            future_idx = np.random.randint(idx, len(transitions))
            future_goal = transitions[future_idx][3]['achieved_goal']
            her_transition = make_her_transition(t, future_goal, reward_fn)
            her_transitions.append(her_transition)

    return her_transitions


def make_her_transition(t, new_goal, reward_fn):
    """Creates new transition from the given one, substitutes the desired
    goal by given new_goal parameter, and recalculates the new reward using the
    given reward_fn
    """

    her_t = copy.deepcopy(t)
    her_t[0]['desired_goal'] = new_goal
    her_t[3]['desired_goal'] = new_goal
    her_t[1] = reward_fn(her_t[3]['achieved_goal'],
                         her_t[3]['desired_goal'], None)
    her_t[0] = flatten_state_dict_for_model(her_t[0])
    her_t[3] = flatten_state_dict_for_model(her_t[3])
    return her_t
