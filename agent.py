import time

import gymnasium as gym
from math import cos
import numpy as np
from abc import abstractmethod

import logger


class Agent:
    def __init__(self,
                 environment: gym.Env,
                 log: bool,
                 technique: int,
                 version: str,
                 learning_rate: float = 0.95,
                 discount: float = 0.95,
                 learning_decay: float = 0.9,
                 learning_min: float = 0.1,
                 epsilon: float = 1,
                 epsilon_min: float = 0.1,
                 episode_count: int = 1000,
                 regression_lambda: float = 0.015,
                 ):
        self.env = environment
        self.discount = discount
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.learning_min = learning_min
        self.epsilon = epsilon
        self.epsilon_decay = epsilon / episode_count
        self.epsilon_min = epsilon_min
        self.episode_count = episode_count
        self.regression_lambda = regression_lambda
        self.technique = technique

        self.version = version
        self.log = log

        # statics
        self.mid_point = -0.3
        self.force = 0.001
        self.gravity = 0.0025
        self.pos_num = 15
        self.right_end = 0.6
        self.left_end = -1.2
        self.vel_num = 15
        self.maximum_vel = 0.07
        self.minimum_vel = -0.07

        self.pos_space = np.linspace(self.left_end, self.right_end, self.pos_num)
        self.vel_space = np.linspace(self.minimum_vel, self.maximum_vel, self.vel_num)

        # features
        self.features_num = 2
        self.features = [
            # self.acceleration_feature,
            self.dist_to_end_feature,
            # self.next_velocity,
            self.dist_to_mid_feature
        ]
        self.weights = [0, 0]
        self.all_episodes = []
        self.all_values = []

    def next_velocity(self, state, action):
        position, velocity = state
        next_vel = velocity + (action - 1) * self.force - cos(3 * position) * self.gravity
        if next_vel <= -0.7:
            next_vel = -0.7
        elif next_vel >= 0.7:
            next_vel = 0.7
        return next_vel

    def next_position(self, state, action):
        position, _ = state
        next_vel = self.next_velocity(state, action)
        next_pos = next_vel * position
        if next_pos <= -1.2:
            next_pos = -1.2
        elif next_pos >= 0.6:
            next_pos = 0.6
        return next_pos

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decay_learning_rate(self):
        self.learning_rate = max(self.learning_min, self.learning_rate * self.learning_decay)

    @staticmethod
    def calculate_acceleration(curr_vel, next_vel):
        return abs(curr_vel - next_vel)

    def calculate_l_weight(self, index, diff, feature_val):
        self.weights[index] = self.weights[index] + self.learning_rate * diff * feature_val

    def calculate_l1_weight(self, index, diff, feature_val):
        if self.weights[index] >= 0:
            self.weights[index] = (self.weights[index] - self.regression_lambda) \
                                  + self.learning_rate * diff * feature_val
        else:
            self.weights[index] = (self.weights[index] + self.regression_lambda) \
                                  + self.learning_rate * diff * feature_val

    def calculate_l2_weight(self, index, diff, feature_val):
        self.weights[index] = (self.weights[index] - 2 * self.regression_lambda * self.weights[index]) \
                              + self.learning_rate * diff * feature_val

    def calculate_weight(self, i, diff, feature_value):
        if self.technique == 3 or self.technique == 0:
            self.calculate_l_weight(i, diff, feature_value)
        elif self.technique == 1:
            self.calculate_l1_weight(i, diff, feature_value)
        elif self.technique == 2:
            self.calculate_l2_weight(i, diff, feature_value)

    def acceleration_feature(self, state, action):
        _, curr_vel = state
        next_vel = self.next_velocity(state, action)
        acceleration = self.calculate_acceleration(curr_vel, next_vel)
        if abs(acceleration) <= 0.0001:
            return (0.0001 - abs(acceleration)) * -10
        return acceleration * 10

    def dist_to_end_feature(self, state, action):
        pos, vel = state
        next_pos = self.next_position(state, action)
        if vel <= 0:
            return 1 / abs(self.left_end - next_pos)
        return 1 / abs(self.right_end - next_pos)

    def dist_to_mid_feature(self, state, action):
        next_pos = self.next_position(state, action)
        dist_to_mid = next_pos - self.mid_point
        return abs(dist_to_mid)

    def velocity_feature(self, state, action):
        _, vel = state
        next_vel = self.next_velocity(state, action) * 5
        next_pos = self.next_position(state, action)
        dist_to_mid = next_pos - self.mid_point
        if dist_to_mid > 0:
            return next_vel
        return next_vel * -1

    @abstractmethod
    def get_qvalue(self, state, action):
        pass

    @abstractmethod
    def update_values(self, curr_state, action, reward, terminated, dynamic_val):
        pass

    def find_max_q_value(self, state):
        return max([self.get_qvalue(state, action) for action in range(self.env.action_space.n)])

    def choose_action(self, curr_state, test_mode=False):
        if test_mode:
            return int(np.argmax([self.get_qvalue(curr_state, action)
                                  for action in range(self.env.action_space.n)]))
        if np.random.random() < self.epsilon:
            self.decay_epsilon()
            return self.env.action_space.sample()
        else:
            return int(np.argmax([self.get_qvalue(curr_state, action)
                                  for action in range(self.env.action_space.n)]))

    def train_agent(self, update_policy):
        if update_policy == 'SARSA':
            self.sarsa_train()
        elif update_policy == 'SARSA-MAX':
            self.sarsa_max_train()

    def sarsa_train(self):
        print('IN SARSA')
        terminated_num = 0
        first_terminated = 0
        pre = time.time()
        for episode in range(self.episode_count):
            curr_state, _ = self.env.reset()
            curr_action = self.choose_action(curr_state=curr_state)
            done = False
            final_reward = 0
            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(curr_action)
                final_reward += reward
                next_action = self.choose_action(curr_state)
                next_state_value = self.get_qvalue(state=curr_state, action=next_action)
                self.update_values(curr_state=curr_state, action=curr_action, reward=int(float(reward)),
                                   terminated=terminated, dynamic_val=next_state_value)
                done = terminated or truncated
                curr_state = next_state
                curr_action = next_action
                if terminated:
                    if terminated_num == 0:
                        first_terminated = episode + 1
                    terminated_num += 1
            self.all_episodes.append(list(self.weights))
            self.all_values.append(final_reward)
            self.decay_learning_rate()
            print(f'Finished episode {episode}')
        if self.log:
            logger.log_terminated(first_terminated, terminated_num, self.episode_count, self.version, time.time() - pre)

    def sarsa_max_train(self):
        terminated_num = 0
        first_terminated = 0
        pre = time.time()
        for episode in range(self.episode_count):
            curr_state, _ = self.env.reset()
            done = False
            final_reward = 0
            while not done:
                action = self.choose_action(curr_state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                max_q_value = self.find_max_q_value(next_state)
                self.update_values(curr_state=curr_state, action=action, reward=int(float(reward)),
                                   terminated=terminated, dynamic_val=max_q_value)
                done = terminated or truncated
                curr_state = next_state
                final_reward += reward
                if terminated:
                    if terminated_num == 0:
                        first_terminated = episode + 1
                    terminated_num += 1
            print(f'Finished episode {episode}')
            self.all_episodes.append(list(self.weights))
            self.all_values.append(final_reward)
            self.decay_learning_rate()
        if self.log:
            logger.log_terminated(first_terminated, terminated_num, self.episode_count, self.version,
                                  time.time() - pre)

