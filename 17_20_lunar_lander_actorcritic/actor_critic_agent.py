import torch
from actor_critic_network import ActorCriticNet

class Agent():
    def __init__(self, load_checkpoint, checkpoint_file, state_space, n_actions=4, lr=0.0001, gamma=0.99):
        self.ac_net = ActorCriticNet(state_space, n_actions, lr, checkpoint_file)
        self.gamma = gamma

        if load_checkpoint:
            self.ac_net.load_checkpoint()
            # set the network to eval mode
            self.ac_net.eval()

    def choose_action(self, state):
        state = torch.tensor([state]).to(self.ac_net.device)
        action_scores, value_est = self.ac_net(state)
        action_probs = torch.softmax(action_scores, dim=1)
        # transform action probs in a categorical distribution, to sample a discrete action
        action_probs = torch.distributions.Categorical(action_probs)
        action = action_probs.sample()
        action_log_prob = action_probs.log_prob(action)

        return action.item(), action_log_prob, value_est

    def learn(self, state, action_log_prob, reward, state_, value_s):
        self.ac_net.optimizer.zero_grad()
        state_ = torch.tensor([state_]).to(self.ac_net.device)
        _, value_s_ = self.ac_net(state_)
        delta = reward + self.gamma * value_s_ - value_s

        actor_loss = - delta * action_log_prob
        critic_loss = torch.square(delta)

        actor_loss.backward(retain_graph=True)
        critic_loss.backward()

    def save_model(self):
        self.ac_net.save_checkpoint()
