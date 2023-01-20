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
from agent import Agent

env = gym.make("MountainCar-v0", max_episode_steps=1000)
observation, _ = env.reset(seed=42)
print(f'initial state --- OBS :: {observation}')
#
mc_agent = Agent(environment=env, initial_position=observation)
mc_agent.train_agent()
env.close()
env = gym.make("MountainCar-v0", max_episode_steps=2000, render_mode='human')
truncated = False
terminated = False

for _ in range(3):
    observation, info = env.reset()
    done = False
    while not done:
        action = mc_agent.choose_action(observation, True)
        observation, reward, terminated, truncated, _ = env.step(action)
        print(f'new data: observation :: {observation}, reward :: {reward}, '
              f'terminated :: {terminated}, truncated :: {truncated}')
        done = terminated or truncated
env.close()
