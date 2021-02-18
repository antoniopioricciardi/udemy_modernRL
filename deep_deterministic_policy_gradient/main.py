import os
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from ddpg_agent import DDPGAgent


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
    plt.title('Average of the previous %d scores' %(n_to_consider))
    plt.savefig(figure_file)


if __name__ == '__main__':
    np.random.seed(0)  # helps with stability
    load_checkpoint = True
    paths = ['plots', 'videos', 'models']
    for path_name in paths:
        if not os.path.exists(path_name):
            os.mkdir(path_name)

    n_games = 1000
    n_states = 8
    n_actions = 2
    mem_size = 10**6
    batch_size = 64
    n_hid1 = 400
    n_hid2 = 300
    lr_alpha = 1e-4
    lr_beta = 1e-3
    gamma = 0.99
    tau = 0.99

    fname = 'lunarlandercontinuous_ngames' + str(n_games) + '_memsize' + str(mem_size) + '_batchsize' + str(batch_size) + '_nhid1' + str(n_hid1)\
        + '_nhid2' + str(n_hid2) + '_lralpha' + str(lr_alpha) + '_lrbeta' + str(lr_beta) + '_gamma' + str(gamma) +\
                '_tau' + str(tau)

    figure_file = 'plots/' + fname + '.png'
    checkpoint_file = 'models/' + fname

    agent = DDPGAgent(load_checkpoint, n_states, n_actions, checkpoint_file, mem_size, batch_size, n_hid1, n_hid2, lr_alpha, lr_beta,
                      gamma, tau)

    env = gym.make('LunarLanderContinuous-v2')
    if not load_checkpoint:
        scores = []
        n_to_consider = 100  # number of previous score to consider in the avg
        best_score = env.reward_range[0]
        for i in range(n_games):
            done = False
            score = 0
            agent.noise.reset()
            obs = env.reset()
            while not done:
                action = agent.choose_action(obs)
                obs_, reward, done, info = env.step(action)
                agent.store_transitions(obs, action, reward, obs_, done)
                agent.learn()
                score += reward
                obs = obs_
            scores.append(score)

            avg_score = np.mean(scores[-n_to_consider:])
            if score > best_score:
                best_score = score
                agent.save_models()
            #if i > 0 and i % n_to_consider == 0:
            print('Epoch %d, score %.3f - %d games avg: %.3f' % (i, score, n_to_consider, avg_score))


        avg_score = np.mean(scores[-n_to_consider:])
        print('Epoch %d, score %.3f - %d games avg: %.3f' % (n_games, score,n_to_consider, avg_score))
        if score > best_score:
            best_score = score
            agent.save_models()

        plot_scores(scores, n_to_consider, figure_file)

    else:
        env = wrappers.Monitor(env, 'videos', video_callable=lambda episode_id: True, force=True)  # force overwrites previous video
        figure_file = 'plots/' + fname + '_eval' + '.png'

        n_games = 10
        n_to_consider = 5
        assert n_games >= n_to_consider
        scores = []
        for i in range(n_games):
            done = False
            obs = env.reset()
            score = 0
            while not done:
                action = agent.choose_action(obs)
                obs_, reward, done, info = env.step(action)
                obs = obs_
                score += reward
            scores.append(score)

            #if i > 0 and i % 100 == 0:
            avg_score = np.mean(scores[-n_to_consider:])
            print('Epoch %d, score %.3f - %d games avg: %.3f' % (i, score, n_to_consider, avg_score))
                # save_check 'models/' + fname

        avg_score = np.mean(scores[-n_to_consider:])
        print('%d games avg: %.3f' % (n_to_consider, avg_score))

        plot_scores(scores, n_to_consider, figure_file)
