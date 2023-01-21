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
from sarsa_agent import SarsaAgent
import logger

env = gym.make("MountainCar-v0", max_episode_steps=1000)
print(env.spec)
observation, _ = env.reset(seed=44)
print(f'initial state --- OBS :: {observation}')

# choose from SarsaMaxAgent and SarsaAgent
log = False
mc_agent = SarsaAgent(environment=env, initial_position=observation, log=log, regression_level=2)
mc_agent.train_agent()
if log:
    logger.plot_weights(mc_agent, 'Sarsa')
env.close()
env = gym.make("MountainCar-v0", max_episode_steps=200, render_mode='human')
truncated = False
terminated = False

for _ in range(20):
    rand_seed = random.randint(1, 10000)
    print(f'seed = {rand_seed}')
    observation, info = env.reset(seed=rand_seed)
    print(f'initial state --- OBS :: {observation}')
    done = False
    while not done:
        action = mc_agent.choose_action(observation, True)
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
env.close()
