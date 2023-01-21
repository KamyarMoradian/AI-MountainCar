import matplotlib.pyplot as plt
import os


def plot_weights(agent, algo) -> None:
    path = os.path.join('./logs/plots', algo, str(agent.regression_level))
    if not os.path.exists(path):
        os.makedirs(path)

    x = [i for i in range(agent.episode_count)]

    for i in range(agent.features_num):
        plt.title(f'Weight {i + 1}')
        plt.xlabel('Time step')
        plt.ylabel('Value')  # noqa

        explore_history = [y[i] for y in agent.all_episodes]
        print(explore_history)

        plt.plot(x, explore_history, label=f'Weight {i}', color='tab:blue')

        plt.legend()
        plt.savefig(path + f'/weight_{i}.png')
        plt.close()


def log_terminated(first_terminated, terminated_count, episode_count, algo, regression_level):
    path = os.path.join(f'.\\logs\\terminated\\{regression_level}\\')
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + algo + '.txt', 'w') as fh:
        fh.writelines([f'Number of episodes = {episode_count}\n',
                       f'Number of terminated episodes = {terminated_count}\n',
                       f'First time terminated = {first_terminated}\n'])
