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
import random

import gymnasium as gym
from sarsa_max_agent import SarsaMaxAgent

env = gym.make("MountainCar-v0", max_episode_steps=1000)
print(env.spec)
observation, _ = env.reset(seed=46)
print(f'initial state --- OBS :: {observation}')
#
mc_agent = SarsaMaxAgent(environment=env, initial_position=observation)
mc_agent.train_agent()
env.close()
env = gym.make("MountainCar-v0", max_episode_steps=200, render_mode='human')
truncated = False
terminated = False

for _ in range(20):
    rand_seed = random.randint(30, 2000)
    print(f'seed = {rand_seed}')
    observation, info = env.reset(seed=rand_seed)
    done = False
    while not done:
        action = mc_agent.choose_action(observation, True)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
env.close()
