import gym
import numpy as np
import matplotlib.pyplot as plt
from cartpole_qlearning_control import Agent


def plot_scores(scores):
    avg_scores = []
    x = []
    for t in range(len(scores)):
        if t < 100:
            avg_scores.append(np.mean(scores[0: t + 1]))
        else:
            avg_scores.append(np.mean(scores[t - 100: t]))
        x.append(t)
    plt.plot(x, avg_scores)
    plt.title('Average of the previous 100 scores')
    plt.show()


def digitize_state(raw_states, states_bins):
    states = []
    for k in range(len(bins)):
        states.append(np.digitize(raw_states[k], states_bins[k]))
    return tuple(states)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    state_cart_pos_bins = np.linspace(-2.4, 2.4, num=10)  # 5 will be the near-zero state (-41.8, 41.8, num=10)
    state_cart_vel_bins = np.linspace(-3.4, 3.4, num=10)
    state_pole_angle_bins = np.linspace(-0.205, 0.205, num=10)
    state_pole_vel_bins = np.linspace(-3.4, 3.4, num=10)

    bins = [state_cart_pos_bins, state_cart_vel_bins, state_pole_angle_bins, state_pole_vel_bins]

    state_space = []
    for cart_pos in range(len(state_cart_pos_bins) + 1):
        for cart_vel in range(len(state_cart_vel_bins) + 1):
            for pole_angle in range(len(state_pole_angle_bins) + 1):
                for pole_vel in range(len(state_pole_vel_bins) + 1):
                    state_space.append((cart_pos, cart_vel, pole_angle, pole_vel))

    n_actions = 2
    n_games = 50000
    alpha = 0.1
    gamma = 0.99
    epsilon = 1
    epsilon_min = 0.01
    epsilon_decr = 2/n_games
    agent = Agent(n_actions, state_space, alpha, gamma, epsilon, epsilon_min, epsilon_decr)

    scores = []
    for i in range(n_games):
        done = False
        score = 0

        raw_obs = env.reset()
        state = digitize_state(raw_obs, bins)
        # cart_pos = np.digitize(raw_obs[0], state_cart_pos_bins)
        # cart_vel = np.digitize(raw_obs[1], state_cart_vel_bins)
        # pole_angle = np.digitize(raw_obs[2], state_pole_angle_bins)
        # pole_vel = np.digitize(raw_obs[3], state_pole_vel_bins)
        # state = (cart_pos, cart_vel, pole_angle, pole_vel)
        while not done:
            action = agent.choose_action(state)
            raw_obs_, reward, done, info = env.step(action)
            state_ = digitize_state(raw_obs_, bins)
            agent.learn(state, action, reward, state_)
            state = state_
            score += reward

        if i > 0 and i % 5000 == 0:
            print('Episode %d, 100 games avg reward: %.3f, epsilon: %.2f' % (i, np.mean(scores[-100:]), agent.epsilon))
        agent.decrease_epsilon()
        scores.append(score)

    plot_scores(scores)
    # avg_scores = []
    # x = []
    # for t in range(len(scores)):
    #     if t < 100:
    #         avg_scores.append(np.mean(scores[0: t+1]))
    #     else:
    #         avg_scores.append(np.mean(scores[t-100: t]))
    #     x.append(t)
    # plt.plot(x, avg_scores)
    # plt.title('Average of the previous 100 scores')
    # plt.show()

