import gym
import numpy as np

'''
Random agent playing Lunar Lander
'''

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    action_space = [i for i in range(4)]

    scores = []
    n_games = 100
    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()
        while not done:
            # env.render()
            action = np.random.choice(action_space)  # pick random action
            state, reward, done, info = env.step(action)
            score += reward

        #scores.append(score)
        print('Epoch %d, score: %.3f' % (i, score))

        # if i % 10 == 0:
        #     avg_score = np.mean(scores[-100:])
        #     print('Epoch %d, avg score: %.3f' % (i, avg_score))

