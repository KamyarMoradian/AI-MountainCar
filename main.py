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
from fa_agent import FAAgent
from ql_agent import QLAgent

CONFIGURATION = {'SEED': 44,
                 'MAX_EPISODES': 1000,
                 'MAX_STEPS': 400,
                 'VERSION': 'QL',
                 'LOG': False,
                 'TECHNIQUE': 3,
                 'UPDATE_POLICY': 'SARSA',
                 'BASE_ALGORITHM': QLAgent,
                 'STR_BASE_ALGORITHM': 'QL',
                 }


def train_agent():
    env = gym.make("MountainCar-v0", max_episode_steps=1000)
    print(env.spec)
    observation, _ = env.reset(seed=CONFIGURATION['SEED'])

    mc_agent = CONFIGURATION['BASE_ALGORITHM'](environment=env, log=CONFIGURATION['LOG'],
                                               technique=CONFIGURATION['TECHNIQUE'], version=CONFIGURATION['VERSION'])
    mc_agent.train_agent(CONFIGURATION['UPDATE_POLICY'])
    if CONFIGURATION['LOG']:
        logger.plot_weights(mc_agent, CONFIGURATION['VERSION'])
        logger.plot_rewards(mc_agent, CONFIGURATION['VERSION'])
    mc_agent.save_agent()
    env.close()
    return mc_agent


def agent_test(mc_agent):
    env = gym.make("MountainCar-v0", max_episode_steps=CONFIGURATION['MAX_STEPS'], render_mode='human')

    chart_data_reward = []
    chart_data_action = []

    for i in range(15):
        rand_seed = random.randint(1, 10000)
        observation, info = env.reset(seed=rand_seed)
        if CONFIGURATION['LOG']:
            env = logger.video_recorder(env, CONFIGURATION['VERSION'], rand_seed)
        done = False
        while not done:
            action = mc_agent.choose_action(observation, True)
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if i == 0:
                chart_data_reward.append(reward)
                chart_data_action.append(action)
    env.close()

    chart_data_time = [i for i in range(len(chart_data_reward))]

    logger.reward_action_graph(chart_data_time, chart_data_reward, chart_data_action,
                               CONFIGURATION['STR_BASE_ALGORITHM'])


def main():
    mc_agent = train_agent()
    agent_test(mc_agent)


if __name__ == '__main__':
    main()
