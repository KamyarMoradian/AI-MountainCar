import gymnasium as gym
import numpy as np
from math import cos


class Agent:
    def __init__(self,
                 environment: gym.Env,
                 initial_position,
                 learning_rate: float = 0.95,
                 discount: float = 0.95,
                 learning_decay: float = 0.9,
                 learning_min: float = 0.1,
                 epsilon: float = 1,
                 epsilon_min: float = 0.1,
                 episode_count: int = 200
                 ):
        self.env = environment
        self.learning_rate = learning_rate
        self.discount = discount
        self.learning_decay = learning_decay
        self.learning_min = learning_min
        self.epsilon = epsilon
        self.epsilon_decay = epsilon / episode_count
        self.epsilon_min = epsilon_min
        self.episode_count = episode_count

        # statics
        self.mid_point = initial_position[0]
        self.force = 0.001
        self.gravity = 0.0025
        self.right_end = 0.6
        self.left_end = -1.2
        # features
        self.features_num = 3
        self.features = [
            self.dist_to_end_feature,
            self.next_velocity,
            self.dist_to_mid_feature
        ]
        self.weights = [0, 0, 0]

    def next_velocity(self, state, action):
        position, velocity = state
        return velocity + (action - 1) * self.force - cos(3 * position) * self.gravity

    def next_position(self, state, action):
        position, _ = state
        next_vel = self.next_velocity(state, action)
        return next_vel * position

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def decay_learning_rate(self):
        self.learning_rate = max(self.learning_min, self.learning_rate * self.learning_decay)

    @staticmethod
    def calculate_acceleration(curr_vel, next_vel):
        return abs(curr_vel - next_vel)

    def acceleration_feature(self, state, action):
        _, curr_vel = state
        next_vel = self.next_velocity(state, action)
        acceleration = self.calculate_acceleration(curr_vel, next_vel)
        if acceleration <= 0.0001:
            return (0.0001 - acceleration) * -10
        return acceleration

    def dist_to_end_feature(self, state, action):
        pos, vel = state
        next_vel = self.next_velocity(state, action)
        next_pos = self.next_position(state, action)
        interval = self.right_end - self.left_end
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

    def update_all_weights(self, curr_state, action, reward, terminated, next_state):
        max_q_value = self.find_max_q_value(next_state)
        target = reward + self.discount * max_q_value * (not terminated)
        prediction = self.get_qvalue(curr_state, action)
        diff = target - prediction
        for i in range(self.features_num):
            feature_value = self.features[i](curr_state, action)
            self.weights[i] = self.weights[i] + self.learning_rate * diff * feature_value
        self.make_weights_normal()

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

    def train_agent(self):
        terminated = truncated = False
        for episode in range(self.episode_count):
            print(f'episode number {episode} ::: final weights = {self.weights}')
            obs, _ = self.env.reset()
            done = False
            count = 1
            while not done:
                action = self.choose_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                self.update_all_weights(curr_state=obs, action=action, reward=int(float(reward)),
                                        terminated=terminated, next_state=next_obs)
                done = terminated or truncated
                obs = next_obs
                count += 1
            self.decay_learning_rate()
