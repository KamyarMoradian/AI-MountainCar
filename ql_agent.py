import os

import numpy as np

from agent import Agent
import gymnasium as gym


class QLAgent(Agent):
    def __init__(self,
                 environment: gym.Env,
                 log,
                 technique: int,
                 version: str):
        super(QLAgent, self).__init__(environment, log, technique, version)
        self.q_values = [[0 for _ in range(environment.action_space.n)] for _ in range(self.vel_num * self.pos_num)]

    def get_state(self, obs):
        pos, vel = obs
        pos_bin = int(np.digitize(pos, self.pos_space))
        vel_bin = int(np.digitize(vel, self.vel_space))
        return pos_bin * self.pos_num + vel_bin

    def get_qvalue(self, state, action):
        curr_state = self.get_state(state)
        return self.q_values[curr_state][action]

    def update_values(self, curr_state, action, reward, terminated, dynamic_val):
        target = reward + self.discount * dynamic_val * (not terminated)
        prediction = self.get_qvalue(curr_state, action)
        diff = target - prediction
        state = self.get_state(curr_state)
        self.q_values[state][action] = prediction + self.learning_rate * diff

    def save_agent(self):
        directory = os.path.join(f'.\\agents\\')
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + 'QL' + '.txt', 'w') as fh:
            for i in range(self.pos_num * self.vel_num):
                for j in range(self.env.action_space.n):
                    fh.write(str(self.q_values[i][j]) + ' ')
                fh.write('\n')
