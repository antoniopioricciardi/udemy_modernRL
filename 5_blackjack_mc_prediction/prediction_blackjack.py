import numpy as np

class Agent():
    def __init__(self, gamma=0.99):
        self.V = {}
        self.sum_space = [i for i in range(4, 22)]  # 4 - 21 possible scores (range is exclusive)
        self.dealer_show_card_space = [i for i in range(1, 11)]
        self.ace_space = [True, False]  # whether the ace is usable or not
        self.action_space = [0, 1]  # stick (no more cards) or hit (another card)

        self.state_space = []  # (sum,dealer_show_card,usable_ace)
        self.returns = {}
        self.states_visited = {}  # first visit or not to each state
        self.memory = []  # (state, reward) pairs - keeps track of states encountered and reward received
        self.gamma = gamma

        self.init_vals()  # to initialize data structures

    def init_vals(self):
        for total in self.sum_space:
            for card in self.dealer_show_card_space:
                for ace in self.ace_space:
                    self.V[(total, card, ace)] = 0
                    self.returns[(total, card, ace)] = []
                    self.states_visited[(total, card, ace)] = False
                    self.state_space.append((total, card, ace))

    def policy(self, state):
        # VERY greedy policy
        total, _, _ = state  # we are only interested in the total sum
        action = 1 if total < 20 else 0
        return action

    def update_V(self):
        """
        Update state values V according to the return of the agent from that state
        """
        for idt, (state, _) in enumerate(self.memory):
            G = 0
            if not self.states_visited[state]:
                self.states_visited[state] = True
                discount = 1
                # start iterating over the agent's memory starting from this particular instance
                # we are only interested in the reward
                for t, (_, reward) in enumerate(self.memory[idt:]):
                    G += reward * discount
                    discount *= self.gamma
                    self.returns[state].append(G)

        for state, _ in self.memory:
            self.V[state] = np.mean(self.returns[state])

        # reset visited states to prepare for next episode
        for state in self.state_space:
            self.states_visited[state] = False

        self.memory = []






