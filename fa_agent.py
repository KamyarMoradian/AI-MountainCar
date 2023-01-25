import os

from agent import Agent
import gymnasium as gym


class FAAgent(Agent):
    def __init__(self,
                 environment: gym.Env,
                 log,
                 technique: int,
                 version: str):
        super(FAAgent, self).__init__(environment, log, technique, version)

    def make_weights_normal(self):
        weights_sum = sum([abs(w) for w in self.weights])
        for i in range(self.features_num):
            self.weights[i] /= weights_sum

    def get_qvalue(self, state, action):
        q_value = 0
        for i in range(self.features_num):
            feature_value = self.features[i](state, action)
            q_value += self.weights[i] * feature_value
        return q_value

    def update_values(self, curr_state, action, reward, terminated, dynamic_val):
        target = reward + self.discount * dynamic_val * (not terminated)
        prediction = self.get_qvalue(curr_state, action)
        diff = target - prediction
        for i in range(self.features_num):
            feature_value = self.features[i](curr_state, action)
            self.calculate_weight(i, diff, feature_value)
        if self.technique == 3:
            self.make_weights_normal()

    def save_agent(self):
        directory = os.path.join(f'.\\agents\\')
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + 'FA' + '.txt', 'w') as fh:
            fh.writelines([f'Weight {i} = {self.weights[i]}\n' for i in range(len(self.weights))])
