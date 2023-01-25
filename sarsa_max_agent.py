import time

import gymnasium as gym

import logger
from agent import Agent


class SarsaMaxAgent(Agent):
    def __init__(self,
                 environment: gym.Env,
                 log: bool,
                 technique: int,
                 version: str):
        super(SarsaMaxAgent, self).__init__(environment, log, technique, version)

    def train_agent(self):
        terminated_num = 0
        first_terminated = 0
        max_q_value = 0
        pre = time.time()
        for episode in range(self.episode_count):
            curr_state, _ = self.env.reset()
            done = False
            final_reward = 0
            while not done:
                action = self.choose_action(curr_state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                max_q_value = self.find_max_q_value(next_state)
                self.update_weights(curr_state=curr_state, action=action, reward=int(float(reward)),
                                              terminated=terminated, dynamic_val=max_q_value)
                done = terminated or truncated
                curr_state = next_state
                final_reward += reward
                if terminated:
                    if terminated_num == 0:
                        first_terminated = episode + 1
                    terminated_num += 1
            self.all_episodes.append(list(self.weights))
            self.all_values.append(final_reward)
            self.decay_learning_rate()
        print('Finished Training')
        if self.log:
            logger.log_terminated(first_terminated, terminated_num, self.episode_count, self.version, time.time() - pre)
