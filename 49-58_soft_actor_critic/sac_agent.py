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

        self.actor = ActorNetwork(n_states, n_actions, n_hid1, n_hid2, self.max_action, lr, checkpoint_file, name='_actor')
        self.critic_1 = CriticNetwork(n_states, n_actions, n_hid1, n_hid2, lr, checkpoint_file, name='_crtic1')
        self.critic_2 = CriticNetwork(n_states, n_actions, n_hid1, n_hid2, lr, checkpoint_file, name='_crtic2')

        self.value_net = ValueNetwork(n_states, n_hid1, n_hid2, lr, checkpoint_file, name='_value')
        self.target_value_net = ValueNetwork(n_states, n_hid1, n_hid2, lr, checkpoint_file, name='_value_target')

        # tau=1 performs an exact copy of the networks to the respective targets
        # self.update_network_parameters(tau=1)
        self.update_network_parameters(self.value_net, self.target_value_net, tau=1)
        # self.update_network_parameters_phil(tau=1)

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
        return actions.cpu().detach().numpy()[0]

    def learn_phil(self):
        if self.memory.mem_counter < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).to(self.critic_1.device)
        done = torch.tensor(done).to(self.critic_1.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.critic_1.device)
        state = torch.tensor(state, dtype=torch.float).to(self.critic_1.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic_1.device)

        value = self.value_net(state).view(-1)
        value_ = self.target_value_net(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparametrize=False)
        # actions, log_probs = self.actor.sample_mvnormal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value_net.optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * (F.mse_loss(value, value_target))
        value_loss.backward(retain_graph=True)
        self.value_net.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparametrize=True)
        # actions, log_probs = self.actor.sample_mvnormal(state, reparameterize=False)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.reward_scale * reward + self.gamma * value_
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        self.update_network_parameters(self.value_net, self.target_value_net)
        # self.update_network_parameters_phil()

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        # state_batch, action_batch, reward_batch, new_state_batch, done_batch = self.sample_transitions()
        state_batch, action_batch, reward_batch, new_state_batch, done_batch = \
            self.memory.sample_buffer(self.batch_size)

        reward_batch = torch.tensor(reward_batch, dtype=torch.float).to(self.critic_1.device)
        done_batch = torch.tensor(done_batch).to(self.critic_1.device)
        new_state_batch = torch.tensor(new_state_batch, dtype=torch.float).to(self.critic_1.device)
        state_batch = torch.tensor(state_batch, dtype=torch.float).to(self.critic_1.device)
        action_batch = torch.tensor(action_batch, dtype=torch.float).to(self.critic_1.device)

        '''Compute Value Network loss'''
        self.value_net.optimizer.zero_grad()
        val = self.value_net(state_batch).view(-1)
        val_ = self.target_value_net(new_state_batch).view(-1)
        val_[done_batch] = 0.0

        actions, log_probs = self.actor.sample_normal(state_batch, reparametrize=False)
        log_probs = log_probs.view(-1)
        q1 = self.critic_1(state_batch, actions)# action_batch)
        q2 = self.critic_1(state_batch, actions)# action_batch)
        q = torch.min(q1, q2).view(-1)
        value_target = q - log_probs
        value_loss = 0.5 * F.mse_loss(val, value_target)

        self.value_net.optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.value_net.optimizer.step()
        # val = val - q + log_prob

        '''Compute Actor loss'''
        self.actor.optimizer.zero_grad()
        # here we need to reparametrize and thus use rsample to make the distribution differentiable
        # because the log prob of the chosen action will be part of our loss.
        actions, log_probs = self.actor.sample_normal(state_batch, reparametrize=True)
        log_probs = log_probs.view(-1)
        q1 = self.critic_1(state_batch, actions)
        q2 = self.critic_2(state_batch, actions)
        q = torch.min(q1, q2).view(-1)
        actor_loss = log_probs - q
        actor_loss = torch.mean(actor_loss)

        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        '''Compute Critic loss'''
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        val_ = self.target_value_net(new_state_batch).view(-1)  # value for the critic update
        val_[done_batch] = 0.0
        q_hat = self.reward_scale * reward_batch + self.gamma * val_
        q1_old_policy = self.critic_1(state_batch, action_batch).view(-1)
        q2_old_policy = self.critic_2(state_batch, action_batch).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters(self.value_net, self.target_value_net, self.tau)
        # self.update_network_parameters_phil()

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

    def update_network_parameters_phil(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value_net.named_parameters()
        value_params = self.value_net.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value_net.load_state_dict(value_state_dict)