import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TD3CriticNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hid1, n_hid2, lr, checkpoint_file, name):
        super(TD3CriticNetwork, self).__init__()

        input_len = n_states + n_actions
        self.fc1 = nn.Linear(input_len, n_hid1)
        self.fc2 = nn.Linear(n_hid1, n_hid2)
        self.fc3 = nn.Linear(n_hid2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.checkpoint_file = checkpoint_file + name

    def forward(self, state, action):

        # as specified in the TD3 paper, we concatenate inputs
        x = torch.cat((state, action), -1).float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_checkpoint(self):
        print('.........Saving checkpoint.........')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('.........Loading checkpoint.........')
        if self.device.type == 'cpu':
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))


class TD3ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hid1, n_hid2, lr, checkpoint_file, name='actor'):
        super(TD3ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(n_states, n_hid1)
        self.fc2 = nn.Linear(n_hid1, n_hid2)
        self.fc3 = nn.Linear(n_hid2, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.checkpoint_file = checkpoint_file + name

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # tanh to bind actions range to +-1. If actions range were greater than +-1, we would multiply
        # the result of this operations by max_action
        x = torch.tanh(self.fc3(x))
        return x

    def save_checkpoint(self):
        print('.........Saving checkpoint.........')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('.........Loading checkpoint.........')
        if self.device.type == 'cpu':
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))
