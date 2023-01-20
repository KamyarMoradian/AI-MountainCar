from agent import Agent
import gymnasium as gym


class SarsaMaxAgent(Agent):
    def __init__(self,
                 environment: gym.Env,
                 initial_position, ):
        super(SarsaMaxAgent, self).__init__(environment, initial_position)

    def train_agent(self):
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
            print(f'episode number {episode} ::: final weights = {self.weights}')
            self.decay_learning_rate()
