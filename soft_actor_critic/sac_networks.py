import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Normal


class ValueNetwork(nn.Module):
    def __init__(self, n_states, n_hid1, n_hid2, lr, checkpoint_file, name):
        super(ValueNetwork, self).__init__()

        self.fc1 = nn.Linear(n_states, n_hid1)
        self.fc2 = nn.Linear(n_hid1, n_hid2)
        self.fc3 = nn.Linear(n_hid2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.checkpoint_file = checkpoint_file + name

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if self.device.type == 'cpu':
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hid1, n_hid2, lr, checkpoint_file, name):
        super(CriticNetwork, self).__init__()

        input_len = n_states + n_actions
        self.fc1 = nn.Linear(input_len, n_hid1)
        self.fc2 = nn.Linear(n_hid1, n_hid2)
        self.fc3 = nn.Linear(n_hid2, 1)  # q1 layer

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.checkpoint_file = checkpoint_file + name

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # q1
        return x

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if self.device.type == 'cpu':
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hid1, n_hid2, max_action, lr, checkpoint_file, name):
        super(ActorNetwork, self).__init__()

        self.reparam_noise = 1e-6 # noise parameter for the reparametrization trick

        self.max_action = max_action
        self.fc1 = nn.Linear(n_states, n_hid1)
        self.fc2 = nn.Linear(n_hid1, n_hid2)
        self.fc3_mu = nn.Linear(n_hid2, n_actions)
        self.fc3_sigma = nn.Linear(n_hid2, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.checkpoint_file = checkpoint_file + name

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.fc3_mu(x)
        sigma = self.fc3_sigma(x)
        # we are using 1e-6 instead of 0, because it may throw an error when sampling a distribution with a st.dev of 0.
        sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)  # in paper they set (-20, 1)
        return mu, sigma

    def sample_normal(self, state, reparametrize=True):
        # for the deterministic policy test just return mu instead of action
        mu, sigma = self.forward(state)  # get mu and sigma for the state
        probabilities = Normal(mu, sigma)  # get distribution
        if reparametrize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        # handle action bounds,
        # by taking the tanh activation and multiplying it by the max_action for our environment
        action = torch.tanh(actions)*torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2) + self.reparam_noise)  # need to add reparam_noise because we cannot take the log of 0, and is possible that the square of the action is 1.
        log_probs = log_probs.sum(1, keepdim=True)
        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        if self.device.type == 'cpu':
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))

