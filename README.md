# prioritized-her
Prioritized Hindsight Experience Replay DDPG agent for openAI robotic gym tasks written in PyTorch

Prioritization is currently based on critic networks, as in DQN. Other option (I might add it soon) is to use the actor error instead. These both approaches are just a simple ideas. I believe that no papers about PHER has been published yet.

### Current status

NOTE: The code seems to **not** work 100% properly. Models aren't learning FetchPush tasks, and during training they behave very unstable, doing actions that might seem like they are avoiding being close to the goal. FetchReach seems to be much easier task and is usually leaned after around 10-15k episodes. This codebase will probably be unstable until model successfully learn FetchPush task.

- DDPG
- Experience Replay
- Exlporation Noise and Dynamic Input Normalization
- Hindsight Experience Replay with 'future' and 'final' modes
- PER as an optional mode(in progress)


21.10.2018: I am making this repository public. Please note that for now, this code might now work 100% correct.


### Running the code

I am using 
- Python 3.6
- PyTorch 0.4.1 
- gym 0.10.5
- mujoco-py 1.50.1.56
- MuJoCo Pro version 1.50

Configuration file explanation in progress. 

Learning script: main.py
Presentation script: presentation.py
