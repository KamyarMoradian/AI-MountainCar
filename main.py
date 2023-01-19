"""
AI agent for Mountain Car problem of class Classic Control
    -> observation space: Box(2,) --> a 2-element vector consisting of
        [0]->position(-1.2, 0.6), [1]->velocity(-0.07, 0.07)
    -> action space: Discrete(3) --> 0 -> move left, 1 -> stay, 2 -> move right
    -> reward: -1
    -> truncated condition:
        maximum step: 200
        reward threshold: -110
"""

import gymnasium as gym

env = gym.make("MountainCar-v0", render_mode="human")
observation, _ = env.reset(seed=42)
print(f'initial state --- OBS :: {observation}')

truncated = False
terminated = False

for episode in range(1):
    done = truncated or terminated
    while not done:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, _ = env.step(action)
        print(f'new data: observation :: {observation}, reward :: {reward}, '
              f'terminated :: {terminated}, truncated :: {truncated}')
        done = terminated or truncated
    observation, info = env.reset()
env.close()
