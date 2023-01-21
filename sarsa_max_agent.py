from agent import Agent
import gymnasium as gym
import logger


class SarsaMaxAgent(Agent):
    def __init__(self,
                 environment: gym.Env,
                 log: bool,
                 initial_position,
                 regression_level: int):
        super(SarsaMaxAgent, self).__init__(environment, initial_position, log, regression_level)

    def train_agent(self):
        terminated_num = 0
        first_terminated = 0
        for episode in range(self.episode_count):
            curr_state, _ = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(curr_state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                max_q_value = self.find_max_q_value(next_state)
                self.update_weights(curr_state=curr_state, action=action, reward=int(float(reward)),
                                              terminated=terminated, dynamic_val=max_q_value)
                done = terminated or truncated
                curr_state = next_state
                if terminated:
                    if terminated_num == 0:
                        first_terminated = episode + 1
                    terminated_num += 1
            self.all_episodes.append(list(self.weights))
            self.decay_learning_rate()
        print('Finished Training')
        if self.log:
            logger.log_terminated(first_terminated, terminated_num, self.episode_count, 'SarsaMax', self.regression_level)
