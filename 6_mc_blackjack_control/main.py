import gym
import matplotlib.pyplot as plt
from control_blackjack import Agent


if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    agent = Agent()
    n_episodes = 200000
    # keep track of wins-loses-draws
    win_lose_draw = {-1:0, 0:0, 1:0}  # -1 lost, 0 draw, 1 win. Values are the number of losses, draws and wins
    win_rates = []
    for i in range(n_episodes):
        if i > 0 and i % 1000 == 0:
            pct_win = win_lose_draw[1] / i
            win_rates.append(pct_win)
        if i % 50000 == 0:
            win_rate = win_rates[-1] if win_rates else 0.0  # print the last saved win rate, if there's
            print('startin episode', i, 'win rate', win_rate)

        # start playing
        obs = env.reset()
        done = False
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
            agent.memory.append((obs, action, reward))
            obs = obs_
        agent.update_Q()
        # env gives rewards only at the end of the episode, -1 for losses, 1 for wins, 0 for draws
        win_lose_draw[reward] += 1
    plt.plot(win_rates)
    plt.show()