import gym
import numpy as np


def policy(state):
    return 0 if state < 5 else 1


def update_V(state, reward, state_):
    V[state] += alpha*(reward + gamma*V[state_] - V[state])


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    V = dict()
    alpha = 0.1
    gamma = 0.99
    # discretize state space in 10 states.
    state_space_bins = np.linspace(-0.2094, 0.2094, num=10)  # 5 will be the near-zero state (-41.8, 41.8, num=10)

    # < -0.2094 is 0 and >= 0.2094 is 10
    for state in range(len(state_space_bins) + 1):
        V[state] = 0

    rewards = []
    n_games = 5000
    for game in range(n_games):
        obs_raw = env.reset()[2]  # undiscretized observation
        obs = np.digitize(obs_raw, state_space_bins)
        done = False
        while not done:
            action = policy(obs)
            obs_raw_, reward, done, info = env.step(action)
            obs_ = np.digitize(obs_raw_[2], state_space_bins)
            V[obs] += alpha * (reward + gamma * V[obs_] - V[obs])
            obs = obs_

    for state in V.keys():
        print(state, '- %.3f' % V[state])




