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

import logger
from sarsa_max_agent import SarsaMaxAgent
from sarsa_agent import SarsaAgent

env = gym.make("MountainCar-v0", max_episode_steps=1000)
print(env.spec)
observation, _ = env.reset(seed=44)
print(f'initial state --- OBS :: {observation}')

# choose from SarsaMaxAgent and SarsaAgent
log = True
technique = 3
version = 'FA-v5.1'
mc_agent = SarsaAgent(environment=env, log=log, technique=technique, version=version)
mc_agent.train_agent()
if log:
    logger.plot_weights(mc_agent, version)
    logger.plot_rewards(mc_agent, version)
env.close()
env = gym.make("MountainCar-v0", max_episode_steps=200, render_mode='rgb_array')
truncated = False
terminated = False

for _ in range(15):
    rand_seed = random.randint(1, 10000)
    observation, info = env.reset(seed=rand_seed)
    env = logger.video_recorder(env, version, rand_seed)
    done = False
    while not done:
        action = mc_agent.choose_action(observation, True)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
env.close()
