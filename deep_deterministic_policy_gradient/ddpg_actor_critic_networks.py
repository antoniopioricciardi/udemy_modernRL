import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hid_1, n_hid_2, lr, checkpoint_file, name='critic'):
        super(CriticNetwork, self).__init__()
        self.checkpoint_file = checkpoint_file + name
        # lr is beta in the paper

        # layer norm seems to be better than batch_norm. When copying to target nets, it does not keep trach of the
        # BatchNorm running mean and variance, moreover it depends of batch_size. With LayerNorm we overcome this problem
        self.fc1 = nn.Linear(n_states, n_hid_1)
        self.fc1_bn = nn.LayerNorm(n_hid_1)
        self.fc2 = nn.Linear(n_hid_1, n_hid_2)
        self.fc2_bn = nn.LayerNorm(n_hid_2)

        # layer to handle action values inputs
        self.action_value = nn.Linear(n_actions, n_hid_2)
        self.q = nn.Linear(n_hid_2, 1)  # the critic value

        '''Init weights'''
        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight, -f1, f1)
        nn.init.uniform_(self.fc1.bias, -f1, f1)
        nn.init.uniform_(self.fc2.weight, -f2, f2)
        nn.init.uniform_(self.fc2.bias, -f2, f2)
        f3 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        nn.init.uniform_(self.action_value.weight, -f3, f3)
        nn.init.uniform_(self.action_value.bias, -f3, f3)

        f3 = 3e-3
        nn.init.uniform_(self.q.weight, -f3, f3)
        nn.init.uniform_(self.q.bias, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-2)
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_val = F.relu(self.fc1_bn(self.fc1(state)))
        state_val = self.fc2_bn(self.fc2(state_val))
        # use action values before activating
        action_val = self.action_value(action)
        state_action_val = F.relu(torch.add(state_val, action_val))  # some concatenate instead of adding, adding should be better
        q_val = self.q(state_action_val)
        return q_val

    def save_checkpoint(self):
        print('.........Saving checkpoint.........')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('.........Loading checkpoint.........')
        if self.device.type == 'cpu':
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, n_states, n_actions, n_hid1, n_hid2, lr, checkpoint_file, name='actor'):
        super(ActorNetwork, self).__init__()
        self.checkpoint_file = checkpoint_file + name
        # lr is alpha in the paper
        self.fc1 = nn.Linear(n_states, n_hid1)
        self.fc1_bn = nn.LayerNorm(n_hid1)
        self.fc2 = nn.Linear(n_hid1, n_hid2)
        self.fc2_bn = nn.LayerNorm(n_hid2)
        self.mu = nn.Linear(n_hid2, n_actions)  # the actor output layer

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        f3 = 3e-3
        ''' init weights '''
        nn.init.uniform_(self.fc1.weight, -f1, f1)
        nn.init.uniform_(self.fc1.bias, -f1, f1)
        nn.init.uniform_(self.fc2.weight, -f2, f2)
        nn.init.uniform_(self.fc2.bias, -f2, f2)
        nn.init.uniform_(self.mu.weight, -f3, f3)
        nn.init.uniform_(self.mu.bias, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)  #1e-4)

        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1_bn(self.fc1(state)))
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = torch.tanh(self.mu(x))  # bounds actions in +- 1 (if actions bound were +-2, we simply had to multiply results by 2)
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