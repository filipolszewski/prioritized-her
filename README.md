# prioritized-her
Prioritized Hindsight Experience Replay DDPG agent for openAI robotic gym tasks written in PyTorch

Prioritization is currently based on critic networks, as in DQN. Other option (I might add it soon) is to use the actor error instead. These both approaches are just a simple ideas. I believe that no papers about PHER has been published yet.

### Current status

NOTE: The code seems to **not** work properly. Models aren't learning the FetchReach and FetchPush tasks, and during training they behave very unstable, doing actions that might seem like they are avoiding being close to the goal. This codebase will probably be unstable until I find the bug or an answer to this issue.

- DDPG
- Experience Replay
- Exlporation Noise and Dynamic Input Normalization
- Hindsight Experience Replay with 'future' and 'final' modes
- PER as an optional mode(in progress)
