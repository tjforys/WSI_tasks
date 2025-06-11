import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def basic_reward(state, next_state, done, reward):
    return reward


def fail_penalty_reward(state, next_state, done, reward):
    if done and reward == 0:
        return -1
    if state == next_state:
        return -0.3
    return reward


def move_closer_to_goal_reward(state, next_state, done, reward):
    if reward:
        return reward
    if state < next_state:
        return 0.02
    else:
        return -0.02


def execute_program(reward_system, slipping, episodes):
    env = gym.make('FrozenLake-v1', desc=None, map_name="8x8", is_slippery=slipping,
                                                        # render_mode="human"
                   )
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))
    num_of_ind_runs = 25
    num_episodes = episodes
    averaged_reward = np.zeros(num_episodes)

    discount = 0.9
    learn_rate = 0.8
    epsilon = 0.1

    for _ in range(num_of_ind_runs):
        qtable = np.zeros((state_size, action_size))
        for episode in range(num_episodes):
            env.reset()
            state = 0
            move_cnt = 0
            while True:
                if np.random.uniform(0, 1) < epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.random.choice(np.flatnonzero(qtable[state] == qtable[state].max()))

                new_state, og_reward, done, _, _ = env.step(action)

                reward = reward_system(state, new_state, done, og_reward)

                qtable[state, action] += learn_rate * (reward + discount * np.max(qtable[new_state]) - qtable[state, action])
                state = new_state

                if done or move_cnt > 200:
                    break
                move_cnt += 1
            averaged_reward[episode] += og_reward
    averaged_reward = averaged_reward / (num_of_ind_runs)
    return averaged_reward


if __name__ == "__main__":
    func1 = basic_reward
    func2 = basic_reward
    averaged_reward_base = execute_program(reward_system=func1, slipping=False, episodes=1000)
    averaged_reward = execute_program(reward_system=func2, slipping=True, episodes=10000)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.plot(averaged_reward_base, 'r', label=func1.__name__)
    plt.plot(averaged_reward, 'b', label=func2.__name__)
    plt.legend()
    plt.show()
