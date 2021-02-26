import os
import gym
import pybullet_envs  # register PyBullet enviroments with open ai gym
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from gym import wrappers


def plot_scores(scores, n_episodes_to_consider, figure_file):
    avg_scores = []
    x = []
    for t in range(len(scores)):
        if t < n_episodes_to_consider:
            avg_scores.append(np.mean(scores[0: t + 1]))
        else:
            avg_scores.append(np.mean(scores[t - n_episodes_to_consider: t]))
        x.append(t)
    plt.plot(x, avg_scores)
    plt.title('Average of the previous %d scores' %(n_episodes_to_consider))
    plt.savefig(figure_file)


# seed = (0)
# env.seed(0)
# np.random.seed(0)
# random.seed(0)
# torch.manual_seed(0)

#env_name = 'LunarLanderContinuous-v2'
# env_name = 'BipedalWalker-v3'
env_name = 'HalfCheetahBulletEnv-v0'
env = gym.make(env_name)

n_games = 3000
n_episodes_to_consider = 50

load_checkpoint = True

n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
update_actor_interval = 2
warmup = 1000
mem_size = 10**6
batch_size = 64
n_hid1 = 400
n_hid2 = 300
lr_alpha = 1e-3
lr_beta = 1e-3
gamma = 0.99
tau = 0.005
noise_mean = 0
noise_sigma = 0.1

fname = 'ngames' + str(n_games) + '_memsize' + str(mem_size) + '_batchsize' + str(batch_size) + '_nhid1' + str(n_hid1)\
    + '_nhid2' + str(n_hid2) + '_lralpha' + str(lr_alpha) + '_lrbeta' + str(lr_beta) + '_gamma' + str(gamma) +\
            '_tau' + str(tau)


figure_file = env_name + '/plots/' + fname + '.png'
checkpoint_file = env_name + '/models/' + fname
agent = Agent(load_checkpoint, checkpoint_file, env, n_states, n_actions, update_actor_interval, warmup, mem_size, batch_size,
              n_hid1, n_hid2, lr_alpha, lr_beta, gamma, tau, noise_mean, noise_sigma)
if load_checkpoint:
    agent.load_models()
if __name__=='__main__':
    if not os.path.exists(env_name):
        os.mkdir(env_name)
    paths = ['plots', 'videos', 'models']
    for path_name in paths:
        path = os.path.join(env_name, path_name)
        if not os.path.exists(path):
            os.mkdir(path)

    if load_checkpoint:
        env = wrappers.Monitor(env, env_name + '/videos', video_callable=lambda episode_id: True,
                               force=True)  #  force overwrites previous video
        figure_file = env_name + '/plots/' + fname + '_eval' + '.png'
        agent.actor.eval()
        n_games = 10
        n_episodes_to_consider = 5

    assert n_games >= n_episodes_to_consider
    scores = []
    best_score = env.reward_range[0]  # 0 is the lowest reward
    for i in range(n_games):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            if load_checkpoint:
                action = agent.choose_action_eval(obs)
            else:
                action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            if not load_checkpoint:
                agent.store_transition(obs, action, reward, obs_, done)
                agent.learn()
            obs = obs_
            score += reward
        scores.append(score)

        avg_score = np.mean(scores[-n_episodes_to_consider:])
        if score > best_score and not load_checkpoint:
            best_score = score
            agent.save_models()
        # if i > 0 and i % n_to_consider == 0:
        print('Epoch %d, score %.3f - %d games avg: %.3f' % (i, score, n_episodes_to_consider, avg_score))
        if i > 0 and i % 200 == 0:
            plot_scores(scores, n_episodes_to_consider, figure_file)

    plot_scores(scores, n_episodes_to_consider, figure_file)



