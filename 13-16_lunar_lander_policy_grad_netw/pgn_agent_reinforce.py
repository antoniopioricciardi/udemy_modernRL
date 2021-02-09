import torch
import torch.distributions.categorical as cat
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from policy_gradient_network import PGN

class Agent():
    def __init__(self, state_space, lr, gamma=0.99, n_actions=4):
        self.state_space = state_space
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions

        self.policy = PGN(self.state_space, self.n_actions, self.lr)

        # keep track of memory rewards per episode and agent's memory of log probs of the actions it took
        self.reward_memory = []
        self.probs_memory = []

    def choose_action(self, observation):
        """
        Need to compute ln(pi(At | St, theta))
        :param observation: state returned by the environment after taking an action
        :return:
        """
        # first thing, convert obs in a tensor and in batch format for pytorch compatibility
        obs = torch.Tensor([observation]).to(self.policy.device)
        # use softmax, so that the sum of all the elements in the vector sum up to 1
        probs = F.softmax(self.policy(obs), dim=1)
        # take prob distrib. and feed it into a categorical distr.
        # similar to numpy random.choice, that choices an element based on a certain prob distributoon
        action_probs = torch.distributions.Categorical(probs)
        action = action_probs.sample()
        # log_probs = action_probs.log_prob()
        self.probs_memory.append(action_probs.log_prob(action))
        # need .item() because action is a tensor, does not work with gym
        return action.item()

    def store_rewards(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        self.policy.optimizer.zero_grad()

        # loss = 0
        # # compute return
        # # G_t = R_t+1 + gamma*R_t+2 + gamma**2*R_t+3 ...
        # for idx, _ in enumerate(self.reward_memory):
        #     G = 0
        #     discount = 1
        #     for reward in self.reward_memory[idx:]:
        #         G += discount*reward
        #         discount *= self.gamma
        #     log_prob = self.probs_memory[idx]
        #     # need "-" sign because we are computing gradient ASCENT, pytorch functions computer descent normally (with +).
        #     loss += -G * log_prob

        # G = np.zeros_like(self.reward_memory)
        # for idx in range(len(self.reward_memory)):
        #     G_sum = 0
        #     discount = 1
        #     for reward in self.reward_memory[idx:]:
        #         G_sum += discount*reward
        #         discount *= self.gamma
        #     G[idx] = G_sum

        G = np.zeros_like(self.reward_memory, dtype=np.float64)
        for t in range(len(self.reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(self.reward_memory)):
                G_sum += self.reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        G = torch.tensor(G, dtype=torch.float).to(self.policy.device)

        loss = 0
        for g, logprob in zip(G, self.probs_memory):
            print(g)
            print(logprob)
            print(loss)
            loss += -g * logprob

        # loss = torch.Tensor(loss).to(self.policy.device)
        loss.backward()
        self.policy.optimizer.step()

        self.probs_memory = []
        self.reward_memory = []

