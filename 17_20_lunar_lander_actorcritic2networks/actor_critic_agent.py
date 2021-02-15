import torch
from actor_critic_2networks import *


class Agent():
    def __init__(self, load_checkpoint, checkpoint_file, n_states,  n_actions=4, n_hid_1=256, n_hid_2=256, actor_lr=0.0001, critic_lr=0.001, gamma=0.99):
        self.actor_net = ActorNet(n_states, n_actions, n_hid_1, n_hid_2, actor_lr, checkpoint_file)
        self.critic_net = CriticNet(n_states, n_hid_1, n_hid_2, critic_lr, checkpoint_file)
        self.gamma = gamma

        self.load_checkpoint = load_checkpoint
        if load_checkpoint:
            self.actor_net.load_checkpoint()
            self.critic_net.load_checkpoint()
            # set the network to eval mode
            self.actor_net.eval()
            self.critic_net.eval()

    def choose_action(self, state):
        state = torch.tensor([state]).to(self.actor_net.device)
        action_scores = self.actor_net(state)
        if not self.load_checkpoint:
            value_est = self.critic_net(state)

        action_probs = torch.softmax(action_scores, dim=1)
        # transform action probs in a categorical distribution, to sample a discrete action
        action_probs = torch.distributions.Categorical(action_probs)
        action = action_probs.sample()
        action_log_prob = action_probs.log_prob(action)

        return action.item(), action_log_prob, value_est

    def learn(self, state, action_log_prob, reward, state_, value_s, done):
        self.actor_net.optimizer.zero_grad()
        self.critic_net.optimizer.zero_grad()
        reward = torch.tensor([reward]).to(self.actor_net.device)
        state_ = torch.tensor([state_]).to(self.actor_net.device)
        _, value_s_ = self.critic_net(state_)

        delta = reward + self.gamma * value_s_*(1-int(done)) - value_s

        actor_loss = -delta * action_log_prob
        critic_loss = delta**2

        actor_loss.backward()
        critic_loss.backward()
        # actor_loss.backward(retain_graph=True)
        # critic_loss.backward()
        self.actor_net.optimizer.step()
        self.critic_net.optimizer.step()

    def save_model(self):
        self.actor_net.save_checkpoint()
        self.critic_net.save_checkpoint()
