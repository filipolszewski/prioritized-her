# prioritized-her
Prioritized Hindsight Experience Replay DDPG agent for openAI robotic gym tasks written in PyTorch

Prioritization is currently based on critic network's, as in DQN. Other option (I might add it soon) is to use the actor error instead. These both approaches are just a simple ideas. I believe that no papers about PHER has been published yet.

### Current status

Note: PER is not working currently, it is in progress. DDPG+HER seems to work correctly after series of fixes.

- Deep Deterministic Policy Gradients
- Experience Replay
- Exlporation Noise and Dynamic Input Normalization
- Hindsight Experience Replay with 'future' and 'final' modes
- PER as an optional mode(in progress)


21.10.2018: I am making this repository public. Please note that for now, this codebase will be unstable and I will be adding and refactoring the code for the next month or so.


### Running the code

I am using 
- Python 3.6
- PyTorch 0.4.1 
- gym 0.10.5
- mujoco-py 1.50.1.56
- MuJoCo Pro version 1.50

Configuration file explanation and documentation in progress. 

Learning script: main.py
Presentation script: presentation.py


### Additional notes

The OU noise that is defined in noise.py is not currently used, as it seems that 
it was making the learning that harder. Currently, the action policy is 