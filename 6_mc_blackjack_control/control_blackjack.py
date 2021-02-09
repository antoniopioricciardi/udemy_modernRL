import numpy as np


class Agent():
    def __init__(self, gamma=0.99, epsilon=0.001):
        self.Q = dict()
        self.V = dict()
        self.policy = dict()

        self.sum_space = [i for i in range(4, 22)]  # 4 - 21 possible scores (range is exclusive)
        self.dealer_show_card_space = [i for i in range(1, 11)]  # card the dealer is showing
        self.ace_space = [True, False]  # whether the ace is usable or not
        self.action_space = [0, 1]  # stick (no more cards) or hit (another card)
        self.state_space = []  # (sum,dealer_show_card,usable_ace)
        self.n_actions = len(self.action_space)

        self.returns = dict()
        self.pairs_visited = dict()  # first visit or not to each state
        self.memory = []  # (state, action, reward) triples - keeps track of states encountered and reward received

        self.gamma = gamma
        self.epsilon = epsilon

        self.init_vals()  # to initialize data structures
        self.init_policy()

    def init_vals(self):
        for cards_sum in self.sum_space:
            for dealer_card in self.dealer_show_card_space:
                for is_usable_ace in self.ace_space:
                    state = (cards_sum, dealer_card, is_usable_ace)
                    self.state_space.append(state)
                    for action in self.action_space:
                        self.Q[(state, action)] = 0
                        self.returns[(state, action)] = []
                        self.pairs_visited[(state, action)] = False

    def init_policy(self):
        for state in self.state_space:
            self.policy[state] = [1/self.n_actions for _ in range(self.n_actions)]

    def choose_action(self, state):
        """
        select action at random from the action space, according to probabilities
        dictated by the agent's policy for that state.
        :param state:
        :return:
        """
        action = np.random.choice(self.action_space, p=self.policy[state])
        return action

    def update_Q(self):
        for idt, (state, action, _) in enumerate(self.memory):
            if not self.pairs_visited[(state, action)]:
                self.pairs_visited[(state, action)] = True
                G = 0
                discount = 1
                # start iterating over the agent's memory starting from this particular instance
                # we are only interested in the reward
                for (_, _, reward) in self.memory[idt:]:
                    G += reward * discount
                    discount *= self.gamma
                    self.returns[(state, action)].append(G)

        for state, action, _ in self.memory:
            self.Q[(state, action)] = np.mean(self.returns[(state, action)])
            self.update_policy(state)

        # reset visited states-action pairs to prepare for next episode
        for state_action in self.pairs_visited.keys():
            self.pairs_visited[state_action] = False

        self.memory = []

    def update_policy(self, state):
        """
        create a list of actions which will correspond to the agent's estimate for
        all the actions in the current state.
        Then take argmax to take the maximum action and set probs for every action in the givne state
        according to the formula:
        1 - eps + (eps/n_actions) if a = max_a
        eps/n_actions if a != max_a
        for the max action, and for the non-maximum actions the other formula
        :param state:
        :return:
        """
        actions = [self.Q[(state, a)] for a in self.action_space]
        a_max = np.argmax(actions)
        probs = []  # to keep track of the probabilities for each action
        for action in self.action_space:
            if action == a_max:
                prob = 1 - self.epsilon + (self.epsilon/self.n_actions)
            else:
                prob = self.epsilon/self.n_actions
            probs.append(prob)
        self.policy[state] = probs

