<h1 align="center">
  <img src="https://github.com/filipolszewski/prioritized-her/blob/master/static/img/logo.png" alt="logo" width="256"></br>
  prioritized-her
</h1>

Prioritized Hindsight Experience Replay DDPG agent for openAI robotic gym tasks written in PyTorch

### 21.10.2018: I am making this repository public. Please note that for now, this codebase will be unstable and I will be adding and refactoring the code for the next month or so!

Prioritization is currently based on critic network's, as in DQN. Other option (I might add it soon) is to use the actor error instead. These both approaches are just a simple ideas. I believe that no papers about PHER has been published yet.

### Current status

Note: PER seems to work now, but I got a process crushed with pretty unclear 
error message once. Not sure of it was local, unrelated thing, if you're 
using the code and notice such behaviour, please let me know.

- Deep Deterministic Policy Gradients
- Experience Replay
- Exploration Noise and Dynamic Input Normalization
- Hindsight Experience Replay with 'future' and 'final' modes
- PER as an optional mode
- Success rate evaluation and success rate plotting
- Generating plots that average given N success rate plots with std_dev intervals (see plot_gen directory)
- **Proper refactor of the code (in progress)**

### Running the code

I am using 
- Python 3.6
- PyTorch 0.4.1 
- gym 0.10.5
- mujoco-py 1.50.1.56
- MuJoCo Pro version 1.50

Configuration file explanation and documentation in progress. 

- Learning script: main.py

- Presentation script: presentation.py

### Results

<h1 align="center">
  <img src="https://github.com/filipolszewski/prioritized-her/blob/master/static/img/result_push.png" alt="result" width="700"></br>
</h1>

### Additional notes

- The 'terminal state' information is not used in training - notice that due to the time limit on the Robotic Envs, the agent receives no info, that could make him reason about the next state being terminal. 
If you would use the mask_batch in the train function, during the calculation of critic's loss, the loss would have been biased and this destabilize learning a lot.
OpenAI mention this as well in the HER paper - "We use the discount factor of γ = 0.98 for all transitions including the ones ending an episode"

- Loading the model is only possible for the presentation mode. I don't feel the need for re-training a saved model.

---

> Header icon made by https://github.com/aboutroots with [Freepik](https://www.freepik.com/) from www.flaticon.com :rat:
