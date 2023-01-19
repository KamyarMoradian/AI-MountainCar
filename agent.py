import gymnasium as gym
import numpy as np
from math import cos


class Agent:
    def __init__(self,
                 environment: gym.Env,
                 learning_rate: float = 0.95,
                 discount: float = 0.95,
                 learning_decay: float = 0.9,
                 learning_min: float = 0.1,
                 epsilon: float = 1,
                 epsilon_min: float = 0.05,
                 episode_count: int = 10
                 ):
        self.env = environment
        self.learning_rate = learning_rate
        self.discount = discount
        self.learning_decay = learning_decay
        self.learning_min = learning_min
        self.epsilon = epsilon
        self.epsilon_decay = epsilon / (episode_count / 2)
        self.epsilon_min = epsilon_min
        self.episode_count = episode_count

        # statics
        self.mid_point = -0.5
        self.force = 0.001
        self.gravity = 0.0025
        self.final_pos = 0.5
        # features
        self.features_num = 2
        self.features = [
            self.acceleration_feature,
            self.dist_to_final_feature,
            # self.dist_to_mid_feature,

        ]
        self.weights = [0, 0]

    def after_mid(self, state):
        position, _ = state
        return position > self.mid_point

    def next_velocity(self, state, action):
        position, velocity = state
        return velocity + (action - 1) * self.force - cos(3 * position) * self.gravity

    def next_position(self, state, action):
        position, _ = state
        next_vel = self.next_velocity(state, action)
        return next_vel * position

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon, self.epsilon - self.epsilon_decay)

    @staticmethod
    def calculate_acceleration(curr_vel, next_vel):
        return abs(curr_vel - next_vel)

    def acceleration_feature(self, state, action):
        print('+++ Acceleration Feature')
        _, curr_vel = state
        next_vel = self.next_velocity(state, action)
        acceleration = self.calculate_acceleration(curr_vel, next_vel)
        if acceleration <= 0.0005:
            return -1
        return acceleration

    def dist_to_final_feature(self, state, action):
        print('+++ Dist to final feature')
        next_pos = self.next_position(state, action)
        return abs(self.final_pos - next_pos) * -1

    # def dist_to_mid_feature(self, state, action):
    #     print('+++ Dist to mid feature')
    #     next_pos = self.next_position(state, action)
    #     return abs(next_pos - self.mid_point)

    def get_qvalue(self, state, action):
        print(f'+++ Calculating QValue for action = {action}')
        q_value = 0
        for i in range(self.features_num):
            feature_value = self.features[i](state, action)
            print(f'calculated feature value = {feature_value}')
            q_value += self.weights[i] * feature_value
        return q_value

    def find_max_q_value(self, state):
        return max([self.get_qvalue(state, action) for action in range(self.env.action_space.n)])

    def update_all_weights(self, curr_state, action, reward, terminated, next_state):
        print('+++ Updating weights...')
        print(f'Taken values ::: state = {curr_state}, action = {action}, reward = {reward},'
              f'terminated = {terminated}, next_state = {next_state}')
        max_q_value = self.find_max_q_value(next_state)
        print(f'max calculated q-value = {max_q_value}')
        target = reward + self.discount * max_q_value * (not terminated)
        print(f'calculated target = {target}')
        prediction = self.get_qvalue(curr_state, action)
        print(f'q-value predicted = {prediction}')
        diff = target - prediction
        print(f'calculated diff = {diff}')
        for i in range(self.features_num):
            feature_value = self.features[i](curr_state, action)
            print(f'calculated feature = {feature_value}')
            print(f'weight before updating = {self.weights[i]}')
            self.weights[i] = self.weights[i] + self.learning_rate * diff * feature_value
            print(f'updated weight = {self.weights[i]}')

    def choose_action(self, curr_state):
        print('+++ choosing action')
        if np.random.random() < self.epsilon:
            self.decay_epsilon()
            chosen_action = self.env.action_space.sample()
            print(f'chosen action = {chosen_action}')
            return chosen_action
        else:
            chosen_action = int(np.argmax([self.get_qvalue(curr_state, action)
                                           for action in range(self.env.action_space.n)]))
            print(f'chosen action = {chosen_action}')
            return chosen_action

    def train_agent(self):
        terminated = truncated = False
        for episode in range(self.episode_count):
            obs, _ = self.env.reset()
            done = truncated or terminated
            count = 1
            while not done:
                print(count)
                action = self.choose_action(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                self.update_all_weights(curr_state=obs, action=action, reward=int(float(reward)),
                                        terminated=terminated, next_state=next_obs)
                print(f'new data: observation :: {next_obs}, reward :: {reward}, '
                      f'terminated :: {terminated}, truncated :: {truncated}')
                done = terminated or truncated
                obs = next_obs
                count += 1
        self.env.close()
