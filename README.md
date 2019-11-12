<h1 align="center">
  <img src="https://github.com/filipolszewski/prioritized-her/blob/master/static/img/logo.png" alt="logo" width="256"></br>
  prioritized-her
</h1>

Prioritized Hindsight Experience Replay DDPG agent for openAI robotic gym tasks written in PyTorch

Prioritization is currently based on critic network, as in DQN. Other option would be to use the actor error instead. See last section for research connected to this mechanism - I believe no 'HER+PER' paper has been published yet, and in my opinion there's a lot possibilities to check. If you want to write such paper and use/extend my code, feel free to do so, but please just let me know.

### Current status

I got a process crushed with pretty unclear error message once. Not sure of it was local, unrelated thing, if you're 
using the code and notice such behaviour, please let me know.

I have also lost the license for MuJoCo engine, so I might be unable to help with fixing new issues.

Done:
- Deep Deterministic Policy Gradients
- Experience Replay
- Exploration Noise and Dynamic Input Normalization
- Hindsight Experience Replay with 'future' and 'final' modes
- PER as an optional mode
- Success rate evaluation and success rate plotting
- Generating plots that average given N success rate plots with std_dev intervals (see plot_gen directory)

Not done:
- **Proper refactor of the code (always in progress...)**

### Running the code

I am using 
- Python 3.6
- PyTorch 0.4.1 
- gym 0.10.5
- mujoco-py 1.50.1.56
- MuJoCo Pro version 1.50

- Configure parameters using configuration.json file

- Learning script: main.py

- Presentation script: presentation.py

- For plotting results see scripts in plot_gen directory

### Results

#### FetchPush

<h1 align="center">
  <img src="https://github.com/filipolszewski/prioritized-her/blob/master/static/img/result_push.png" alt="result" width="700"></br>
</h1>

#### FetchReach

<h1 align="center">
  <img src="https://github.com/filipolszewski/prioritized-her/blob/master/static/img/result_reach.png" alt="result" width="700"></br>
</h1>

### Additional notes

- The 'terminal state' information is not used in training - notice that due to the time limit on the Robotic Envs, the agent receives no info, that could make him reason about the next state being terminal. 
If you would use the mask_batch in the train function, during the calculation of critic's loss, the loss would have been biased and this destabilize learning a lot.
OpenAI mention this as well in the HER paper - "We use the discount factor of γ = 0.98 for all transitions including the ones ending an episode"

- Loading the model is only possible for the presentation mode. I don't feel the need for re-training a saved model.

- Fun result: For FetchReach env, agents are learning with fully noisy policy
 used. Try DDPG+HER+PER and epsilon = epsilon_decay = 1.
 
- No tests were done for other environments.
---

### Similar research

- https://arxiv.org/abs/1905.05498 - Bias-Reduced Hindsight Experience Replay with Virtual Goal Prioritization (PER mentioned, only theoretical comparison)
- https://arxiv.org/abs/1810.01363 - Energy-Based Hindsight Experience Prioritization (HER + PER implemented, comparison with EBP was done on 4 environments, HER+PER was weaker in those)

> Header icon made by https://github.com/aboutroots with [Freepik](https://www.freepik.com/) from www.flaticon.com :rat:
