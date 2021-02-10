import os
import gym
import numpy as np
import matplotlib.pyplot as plt

from pgn_agent_reinforce import Agent
from gym import wrappers


def plot_scores(scores, n_to_consider, figure_file):
    avg_scores = []
    x = []
    for t in range(len(scores)):
        if t < n_to_consider:
            avg_scores.append(np.mean(scores[0: t + 1]))
        else:
            avg_scores.append(np.mean(scores[t - n_to_consider: t]))
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
    load_checkpoint = True
    if not os.path.exists('plots'):
        os.mkdir('plots')
    if not os.path.exists('videos'):
        os.mkdir('videos')
    if not os.path.exists('models') and not load_checkpoint:
        os.mkdir('models')

    env = gym.make('LunarLander-v2')
    n_games = 3000
    lr = 0.0005
    gamma = 0.99

    fname = 'REINFORCE_lunarlander_lr' + str(lr) + '_gamma' + str(gamma) + '_ngames' + str(n_games)
    #if not os.path.exists(fname):
    #    os.mkdir(fname)
    figure_file = 'plots/' + fname + '.png'
    checkpoint_file = 'models/' + fname

    agent = Agent(load_checkpoint, checkpoint_file, [8], lr, gamma, 4)
    if not load_checkpoint:
        n_to_consider = 100
        scores = []
        prev_score = -1000
        for i in range(n_games):
            done = False
            obs = env.reset()
            score = 0
            while not done:
                action = agent.choose_action(obs)
                obs_, reward, done, info = env.step(action)
                agent.store_rewards(reward)
                obs = obs_
                score += reward
            scores.append(score)
            agent.learn()

            if i > 0 and i % 100 == 0:
                avg_score = np.mean(scores[-100:])
                print('Epoch %d, 100 games avg: %.3f' % (i, avg_score))
                if avg_score > prev_score:
                    prev_score = avg_score
                    agent.save_model()

        avg_score = np.mean(scores[-100:])
        print('Epoch %d, 100 games avg: %.3f' % (n_games, avg_score))
        if avg_score > prev_score:
            prev_score = avg_score
            agent.save_model()

        plot_scores(scores, n_to_consider, figure_file)

    else:
        env = wrappers.Monitor(env, 'videos', video_callable=lambda episode_id: True, force=True)  # force overwrites previous video
        n_to_consider = 5
        scores = []
        for i in range(n_to_consider):
            done = False
            obs = env.reset()
            score = 0
            while not done:
                action = agent.choose_action(obs)
                obs_, reward, done, info = env.step(action)
                obs = obs_
                score += reward
            scores.append(score)

            if i > 0 and i % 100 == 0:
                avg_score = np.mean(scores[-100:])
                print('Epoch %d, 100 games avg: %.3f' % (i, avg_score))
                    # save_check 'models/' + fname

        avg_score = np.mean(scores[-n_to_consider:])
        print('%d games avg: %.3f' % (n_to_consider, avg_score))

        plot_scores(scores, n_to_consider, figure_file)
