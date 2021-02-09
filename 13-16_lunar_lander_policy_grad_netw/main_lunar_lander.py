import os
import gym
import numpy as np
import matplotlib.pyplot as plt

from pgn_agent_reinforce import Agent


def plot_scores(scores, figure_file):
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
    plt.savefig(figure_file)


if __name__ == '__main__':
    '''
    By looking at their github repo, I can say that first 2 are position in x axis and y axis(hieght)
    other 2 are the x,y axis velocity terms, lander angle and angular velocity,
    left and right left contact points (bool) -- thanks reddit
    '''
    if not os.path.exists('plots'):
        os.mkdir('plots')
    env = gym.make('LunarLander-v2')
    lr = 0.0005
    gamma = 0.99
    agent = Agent([8], lr, 0.99, 4)

    n_games = 3000
    fname = 'REINFORCE_lunarlander_lr' + str(lr) + '_gamma' + str(gamma) + '_ngames' + str(n_games)
    if not os.path.exists(fname):
        os.mkdir(fname)
    figure_file = 'plots/' + fname + '.png'

    scores = []
    done = False
    obs = env.reset()
    for i in range(n_games):
        score = 0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            agent.store_rewards(reward)
            obs = obs_
            score += reward
        agent.learn()

        if i > 0 and i % 100 == 0:
            print('Epoch %d, 100 games avg: %.3f' % (i, np.mean(scores[-100:])))
    plot_scores(scores, figure_file)
