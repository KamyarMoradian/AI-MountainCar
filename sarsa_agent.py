from agent import Agent
import gymnasium as gym
import logger


class SarsaAgent(Agent):
    def __init__(self,
                 environment: gym.Env,
                 initial_position,
                 log,
                 regression_level: int):
        super(SarsaAgent, self).__init__(environment, initial_position, log, regression_level)

    def train_agent(self):
        terminated_num = 0
        first_terminated = 0
        for episode in range(self.episode_count):
            curr_state, _ = self.env.reset()
            curr_action = self.choose_action(curr_state=curr_state)
            done = False
            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(curr_action)
                next_action = self.choose_action(curr_state)
                next_state_value = self.get_qvalue(state=curr_state, action=next_action)
                self.update_weights(curr_state=curr_state, action=curr_action, reward=int(float(reward)),
                                    terminated=terminated, dynamic_val=next_state_value)
                done = terminated or truncated
                curr_state = next_state
                curr_action = next_action
                if terminated:
                    if terminated_num == 0:
                        first_terminated = episode + 1
                    terminated_num += 1
            self.all_episodes.append(list(self.weights))
            self.decay_learning_rate()
        if self.log:
            logger.log_terminated(first_terminated, terminated_num, self.episode_count, 'Sarsa', self.regression_level)
