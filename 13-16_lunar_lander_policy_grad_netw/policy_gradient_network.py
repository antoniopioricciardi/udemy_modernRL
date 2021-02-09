import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PGN(nn.Module):
    def __init__(self, state_space, n_actions, lr):
        super(PGN, self).__init__()

        self.input_layer = nn.Linear(*state_space, 128)  # * means we are going to unpack a list
        self.hid_layer = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        # self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.input_layer(state))
        x = F.relu(self.hid_layer(x))
        x = self.output_layer(x)
        return x




