from sac_networks import *
from replaymemory import ReplayMemory


class Agent:
    def __init__(self, load_checkpoint, checkpoint_file, env, n_states, n_actions,
                 mem_size=10 ** 6, batch_size=256, n_hid1=256, n_hid2=256, lr=3e-4, gamma=0.99,
                 tau=5e-3, reward_scale=2):

        self.load_checkpoint = load_checkpoint

        self.max_action = float(env.action_space.high[0])
        self.low_action = float(env.action_space.low[0])

        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale

        self.memory_counter = 0
        self.memory = ReplayMemory(mem_size, n_states, n_actions)

        self.actor = ActorNetwork(n_states, n_actions, n_hid1, n_hid2, self.max_action, lr, checkpoint_file, name='actor')
        self.critic_1 = CriticNetwork(n_states, n_actions, n_hid1, n_hid2, lr, checkpoint_file, name='crtic1')
        self.critic_2 = CriticNetwork(n_states, n_actions, n_hid1, n_hid2, lr, checkpoint_file, name='crtic2')

        self.value_net = ValueNetwork(n_states, n_hid1, n_hid2, lr, checkpoint_file, name='value')
        self.target_value_net = ValueNetwork(n_states, n_hid1, n_hid2, lr, checkpoint_file, name='value_target')

        # tau=1 performs an exact copy of the networks to the respective targets
        # self.update_network_parameters(tau=1)
        self.update_network_parameters(self.value_net, self.target_value_net, tau=1)

    def store_transition(self, obs, action, reward, obs_, done):
        self.memory.store_transition(obs, action, reward, obs_, done)

    def sample_transitions(self):
        state_batch, action_batch, reward_batch, new_state_batch, done_batch = self.memory.sample_buffer(
            self.batch_size)
        # no need to care about the device, it is the same for all class objects (cuda or cpu is the same despite the class)
        state_batch = torch.tensor(state_batch, dtype=torch.float).to(self.actor.device)
        action_batch = torch.tensor(action_batch, dtype=torch.float).to(self.actor.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float).to(self.actor.device)
        new_state_batch = torch.tensor(new_state_batch, dtype=torch.float).to(self.actor.device)
        done_batch = torch.tensor(done_batch).to(self.actor.device)
        return state_batch, action_batch, reward_batch, new_state_batch, done_batch

    def update_network_parameters(self, network, target_network, tau=None):
        for par, target_par in zip(network.parameters(), target_network.parameters()):
            target_par.data.copy_(tau * par.data + (1 - tau) * target_par.data)

    def choose_action(self, obs):
        obs = torch.tensor([obs], dtype=torch.float).to(self.actor.device)
        actions, _ = self.actor.sample_normal(obs, reparametrize=False)
        return actions.cpu().detatch().numpy()[0]

    def learn(self):
        if self.memory_counter < self.batch_size:
            return
        state_batch, action_batch, reward_batch, new_state_batch, done_batch = self.sample_transitions()

        '''Compute Value Network loss'''
        val = self.value_net(state_batch).view(-1)
        actions, log_probs = self.actor(state_batch, reparametrize=False)
        log_probs = log_probs.view(-1)
        q1 = self.critic_1(state_batch, actions)# action_batch)
        q2 = self.critic_1(state_batch, actions)# action_batch)
        q = torch.min(q1, q2).view(-1)
        value_target = q - log_probs
        value_loss = 0.5 * F.mse_loss(val, value_target)
        # val = val - q + log_prob

        '''Compute Actor loss'''
        # here we need to reparametrize and thus use rsample to make the distribution differentiable
        # because the log prob of the chosen action will be part of our loss.
        actions, log_probs = self.actor.sample_normal(state_batch, reparametrize=True)
        log_probs = log_probs.view(-1)
        q1 = self.critic_1(state_batch, actions)
        q2 = self.critic_2(state_batch, actions)
        q = torch.min(q1, q2).view(-1)
        actor_loss = log_probs - q
        actor_loss = torch.mean(actor_loss)

        '''Compute Critic loss'''
        val_ = self.target_value_net(new_state_batch).view(-1)  # value for the critic update
        val_[done_batch] = 0.0
        val_.view(-1)
        q_hat = self.reward_scale * reward_batch + self.gamma * val_
        q1_old_policy = self.critic_1(state_batch, action_batch)
        q2_old_policy = self.critic_2(state_batch, action_batch)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss

        self.value_net.optimizer.zero_grad()
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()

        value_loss.backward(retain_graph=True)
        self.value_net.optimizer.step()

        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters(self.value_net, self.target_value_net)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value_net.save_checkpoint()
        self.target_value_net.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.value_net.load_checkpoint()
        self.target_value_net.load_checkpoint()
