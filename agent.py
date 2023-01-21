import gymnasium as gym
import numpy as np
from math import cos


class Agent:
    def __init__(self,
                 environment: gym.Env,
                 initial_position,
                 log: bool,
                 regression_level: int,
                 learning_rate: float = 0.95,
                 discount: float = 0.95,
                 learning_decay: float = 0.9,
                 learning_min: float = 0.1,
                 epsilon: float = 1,
                 epsilon_min: float = 0.1,
                 episode_count: int = 100,
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
        self.regression_level = regression_level

        self.log = log

        # statics
        self.mid_point = -0.3
        self.force = 0.001
        self.gravity = 0.0025
        self.right_end = 0.6
        self.left_end = -1.2
        # features
        self.features_num = 4
        self.features = [
            self.acceleration_feature,
            self.dist_to_end_feature,
            self.next_velocity,
            self.dist_to_mid_feature
        ]
        self.weights = [0, 0, 0, 0]
        self.all_episodes = []

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
        if self.regression_level == 3 or self.regression_level == 0:
            self.calculate_l_weight(i, diff, feature_value)
        elif self.regression_level == 1:
            self.calculate_l1_weight(i, diff, feature_value)
        elif self.regression_level == 2:
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

    def get_qvalue(self, state, action):
        q_value = 0
        for i in range(self.features_num):
            feature_value = self.features[i](state, action)
            q_value += self.weights[i] * feature_value
        return q_value

    def find_max_q_value(self, state):
        return max([self.get_qvalue(state, action) for action in range(self.env.action_space.n)])

    def make_weights_normal(self):
        weights_sum = sum([abs(w) for w in self.weights])
        for i in range(self.features_num):
            self.weights[i] /= weights_sum

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

    def update_weights(self, curr_state, action, reward, terminated, dynamic_val):
        target = reward + self.discount * dynamic_val * (not terminated)
        prediction = self.get_qvalue(curr_state, action)
        diff = target - prediction
        for i in range(self.features_num):
            feature_value = self.features[i](curr_state, action)
            self.calculate_weight(i, diff, feature_value)
        if self.regression_level == 3:
            self.make_weights_normal()
