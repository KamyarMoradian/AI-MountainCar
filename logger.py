import gymnasium as gym
import matplotlib.pyplot as plt
import os


def plot_rewards(agent, version):
    print('Plotting rewards...')
    directory = r'.\\logs\\plots\\rewards'
    path = os.path.join(directory, version)
    if not os.path.exists(path):
        os.makedirs(path)

    x = [i for i in range(agent.episode_count)]

    plt.title(f'{version}-rewards')
    plt.xlabel('Time step')
    plt.ylabel('Value')

    explore_history = [y for y in agent.all_values]

    plt.plot(x, explore_history, label=f'reward', color='tab:blue')

    plt.legend()
    plt.savefig(path + f'/reward.png')
    plt.close()


def plot_weights(agent, version) -> None:
    print('Plotting weights...')
    directory = r'.\\logs\\plots\\weights'
    path = os.path.join(directory, version)
    if not os.path.exists(path):
        os.makedirs(path)

    x = [i for i in range(agent.episode_count)]

    for i in range(agent.features_num):
        plt.title(f'Weight {i + 1}')
        plt.xlabel('Time step')
        plt.ylabel('Value')

        explore_history = [y[i] for y in agent.all_episodes]
        print(explore_history)

        plt.plot(x, explore_history, label=f'Weight {i}', color='tab:blue')

        plt.legend()
        plt.savefig(path + f'/weight_{i}.png')
        plt.close()


def log_terminated(first_terminated, terminated_count, episode_count, version, time_taken) -> None:
    directory = os.path.join(f'.\\logs\\terminated')
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, version + '\\')
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + 'result.txt', 'w') as fh:
        fh.writelines([f'Number of episodes = {episode_count}\n',
                       f'Number of terminated episodes = {terminated_count}\n',
                       f'First time terminated = {first_terminated}\n',
                       f'Time Taken = {time_taken}'])


def video_recorder(env, version, seed):
    return gym.wrappers.RecordVideo(env, f'video\\{version}\\{str(seed)}',
                                    name_prefix=version + '_' + 'seed' + str(seed),
                                    episode_trigger=lambda x: x % 6 == 0)


def reward_action_graph(chart_data_time, chart_data_reward, chart_data_action, algo):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(chart_data_time, chart_data_reward,
               chart_data_action, c='r', marker='o')
    ax.set_xlabel('Time')
    ax.set_ylabel('Reward')
    ax.set_zlabel('Action')
    plt.show()
    # save it in an image file
    directory = os.path.join(f'.\\logs\\FinalCharts')
    if not os.path.exists(directory):
        os.makedirs(directory)
    ax.figure.savefig(f'.\\logs\\FinalCharts\\{algo}.png')
