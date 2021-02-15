import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from actor_critic_agent import Agent


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
    load_checkpoint = True

    if not os.path.exists('plots'):
        os.mkdir('plots')
    if not os.path.exists('videos'):
        os.mkdir('videos')
    if not os.path.exists('models') and not load_checkpoint:
        os.mkdir('models')

    lr = 5e-6
    gamma = 0.99
    n_games = 2000
    fname = 'ACTORCRITIC_lunarlander_lr' + str(lr) + '_gamma' + str(gamma) + '_ngames' + str(n_games)
    checkpoint_file = 'models/' + fname
    figure_file = 'plots/' + fname + '.png'

    env = gym.make('LunarLander-v2')
    n_hid_1 = 2048
    n_hid_2 = 1536
    n_states = 8
    n_actions = 4
    agent = Agent(load_checkpoint, checkpoint_file, n_states, n_actions, n_hid_1, n_hid_2, lr, gamma)

    if not load_checkpoint:
        n_to_consider = 100
        prev_score = -1000
        scores = []
        for i in range(n_games):
            score = 0
            done = False
            obs = env.reset()
            while not done:
                action, action_log_prob, value_est = agent.choose_action(obs)
                obs_, reward, done, info = env.step(action)
                agent.learn(obs, action_log_prob, reward, obs_, value_est, done)
                obs = obs_

                score += reward
            scores.append(score)

            # if i > 0 and i % n_to_consider == 0:
            avg_score = np.mean(scores[-n_to_consider:])
            print('Epoch %d, score %.3f - 100 games avg: %.3f' % (i, score, avg_score))
            if avg_score > prev_score:
                prev_score = avg_score
                agent.save_model()

        avg_score = np.mean(scores[-n_to_consider:])
        print('%d games avg: %.3f' % (n_to_consider, avg_score))
        if avg_score > prev_score:
            prev_score = avg_score
            agent.save_model()

        plot_scores(scores, n_to_consider, figure_file)

    else:
        env = wrappers.Monitor(env, 'videos', video_callable=lambda episode_id: True, force=True)  # force overwrites previous video
        figure_file = 'plots/' + fname + '_eval' + '.png'

        n_to_consider = 5
        scores = []
        for i in range(n_to_consider):
            done = False
            obs = env.reset()
            score = 0
            while not done:
                action, _, _ = agent.choose_action(obs)
                obs_, reward, done, info = env.step(action)
                obs = obs_

                score += reward
            scores.append(score)

            # if i > 0 and i % n_to_consider == 0:
            avg_score = np.mean(scores[-n_to_consider:])
            print('Epoch %d, score %.3f - 100 games avg: %.3f' % (i, score, avg_score))
                # save_check 'models/' + fname

        avg_score = np.mean(scores[-n_to_consider:])
        print('%d games avg: %.3f' % (n_to_consider, avg_score))

        plot_scores(scores, n_to_consider, figure_file)
