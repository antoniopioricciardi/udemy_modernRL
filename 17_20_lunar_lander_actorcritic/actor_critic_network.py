import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ActorCriticNet(nn.Module):
    def __init__(self, n_states, n_actions, n_hid_1, n_hid_2, lr, checkpoint_file):
        super(ActorCriticNet, self).__init__()

        self.fc1 = nn.Linear(n_states, n_hid_1)
        self.fc2 = nn.Linear(n_hid_1, n_hid_2)
        self.fc_actor = nn.Linear(n_hid_2, n_actions)
        self.fc_critic = nn.Linear(n_hid_2, 1)

        # maybe declare actor and critic losses
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.checkpoint_file = checkpoint_file

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_scores = self.fc_actor(x)
        value_est = self.fc_critic(x)
        return action_scores, value_est

    def save_checkpoint(self):
        print('.........Saving checkpoint.........')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('.........Loading checkpoint.........')
        if self.device.type == 'cpu':
            self.load_state_dict(torch.load(self.checkpoint_file, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(self.checkpoint_file))
