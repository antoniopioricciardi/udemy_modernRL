import numpy as np

class Agent():
    def __init__(self, n_actions, state_space,  alpha, gamma, epsilon, epsilon_min, epsilon_decr):
        # state space (pole_position, pole_velocity, pole_angle, pole_velocity), all in range [0, 10]
        self.n_actions = n_actions

        self.actions = [i for i in range(self.n_actions)]
        self.state_space = state_space

        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decr = epsilon_decr
        self.gamma = gamma

        self.Q = dict()

        self.initialize_Q()

    def initialize_Q(self):
        for state in self.state_space:
            for action in self.actions:
                self.Q[(state, action)] = 0.0

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            actions = [self.Q[(state, a)] for a in self.actions]
            action = np.argmax(actions)
        return action

    def decrease_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decr

    def learn(self, state, action, reward, state_):
        # Q[s,a] += Q[s,a] + alpha*(r + gamma* max_a_ Q(s_, a_) - Q(s,a))
        # pick max next Q value
        vals_actions_ = [self.Q[(state_, a_)] for a_ in self.actions]
        next_q_val = np.max(vals_actions_)

        q_val = self.Q[(state, action)]

        self.Q[(state, action)] += self.alpha*(reward + self.gamma*next_q_val - q_val)
        # self.decrease_epsilon()



