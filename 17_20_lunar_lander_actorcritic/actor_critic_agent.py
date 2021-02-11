import torch
from actor_critic_network import ActorCriticNet

class Agent():
    def __init__(self, load_checkpoint, checkpoint_file, n_states,  n_actions=4, n_hid_1=256, n_hid_2=256, lr=0.0001, gamma=0.99):
        self.ac_net = ActorCriticNet(n_states, n_actions, n_hid_1, n_hid_2, lr, checkpoint_file)
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

    def learn(self, state, action_log_prob, reward, state_, value_s, done):
        self.ac_net.optimizer.zero_grad()
        reward = torch.tensor([reward]).to(self.ac_net.device)
        state_ = torch.tensor([state_]).to(self.ac_net.device)
        _, value_s_ = self.ac_net(state_)

        delta = reward + self.gamma * value_s_*(1-int(done)) - value_s

        actor_loss = -delta * action_log_prob
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()
        # actor_loss.backward(retain_graph=True)
        # critic_loss.backward()
        self.ac_net.optimizer.step()

    def save_model(self):
        self.ac_net.save_checkpoint()
