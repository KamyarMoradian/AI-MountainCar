from agent import Agent
import numpy as np
import gymnasium as gym


class SarsaMaxAgent(Agent):
    def __init__(self,
                 environment: gym.Env,
                 initial_position, ):
        super(SarsaMaxAgent, self).__init__(environment, initial_position)

    def update_all_weights(self, curr_state, action, reward, terminated, next_state):
        max_q_value = self.find_max_q_value(next_state)
        target = reward + self.discount * max_q_value * (not terminated)
        prediction = self.get_qvalue(curr_state, action)
        diff = target - prediction
        for i in range(self.features_num):
            feature_value = self.features[i](curr_state, action)
            self.weights[i] = self.weights[i] + self.learning_rate * diff * feature_value
        self.make_weights_normal()

    def train_agent(self):
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
