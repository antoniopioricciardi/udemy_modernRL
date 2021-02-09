import gym
from prediction_blackjack import Agent

if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    agent = Agent()
    n_episodes = 500000
    for i in range(n_episodes):
        if i % 50000 == 0:
            print('starting ep.', i)

        obs = env.reset()
        done = False
        while not done:
            action = agent.policy(obs)
            obs_, reward, done, info = env.step(action)
            agent.memory.append((obs, reward))
            obs = obs_
        agent.update_V()
    print(agent.V[(21, 3, True)])  # likely to win, should be high value
    print(agent.V[(4, 1, False)])  # likely to lose, should be low value
