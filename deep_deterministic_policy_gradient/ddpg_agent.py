import torch
import torch.nn.functional as F
import numpy as np

from ddpg_actor_critic_networks import ActorNetwork, CriticNetwork
from replaymemory import ReplayMemory
from ou_action_noise import OUActionNoise


class DDPGAgent():
    def __init__(self, load_checkpoint, n_states, n_actions, checkpoint_file, mem_size=10**6, batch_size=64, n_hid1=400, n_hid2=300,
                 alpha=1e-4, beta=1e-3, gamma=0.99, tau=0.99):
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.actor = ActorNetwork(n_states, n_actions, n_hid1, n_hid2, alpha, checkpoint_file, name='actor')
        self.critic = CriticNetwork(n_states, n_actions, n_hid1, n_hid2, beta, checkpoint_file, name='critic')

        self.actor_target = ActorNetwork(n_states, n_actions, n_hid1, n_hid2, alpha, checkpoint_file, name='actor_target')
        self.critic_target = CriticNetwork(n_states, n_actions, n_hid1, n_hid2, beta, checkpoint_file, name='critic_target')

        self.noise = OUActionNoise(mu=np.zeros(n_actions))
        self.memory = ReplayMemory(mem_size, n_states, n_actions)
        self.update_network_parameters_phil(tau=1)
        if load_checkpoint:
            self.actor.eval()
        self.load_checkpoint = load_checkpoint

    def reset_noise(self):
        self.noise.reset()

    def __copy_param(self, net_param_1, net_param_2, tau):
        # a.copy_(b) reads content from b and copy it to a
        for par, target_par in zip(net_param_1, net_param_2):
            with torch.no_grad():
                val_to_copy = tau * par.weight + (1 - tau) * target_par.weight
                target_par.weight.copy_(val_to_copy)
                if target_par.bias is not None:
                    val_to_copy = tau * par.bias + (1 - tau) * target_par.bias
                    target_par.bias.copy_(val_to_copy)

    def update_network_parameters(self, tau=None):
        # TODO: Controlla equivalenza con metodo Phil
        # during the class initialization we call this method with tau=1, to perform an exact copy of the nets to targets
        # then when we call this without specifying tau, we use the field stored
        if tau is None:
            tau = self.tau

        actor_params = self.actor.children()
        actor_target_params = self.actor_target.children()
        self.__copy_param(actor_params, actor_target_params, tau)

        critic_params = self.critic.children()
        critic_target_params = self.critic_target.children()
        self.__copy_param(critic_params, critic_target_params, tau)

    def choose_action(self, obs):
        # when using layer norm, we do not want to calculate statistics for the forward propagation. Not needed
        # if using batchnorm or dropout
        self.actor.eval()
        obs = torch.tensor(obs, dtype=torch.float).to(self.actor.device)
        # compute actions
        mu = self.actor(obs)
        # add some random noise for exploration
        mu_prime = mu
        if not self.load_checkpoint:
            mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device)
            self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def store_transitions(self, obs, action, reward, obs_, done):
        self.memory.store_transition(obs, action, reward, obs_, done)

    def sample_transitions(self):
        state_batch, action_batch, reward_batch, new_state_batch, done_batch = self.memory.sample_buffer(self.batch_size)
        # no need to care about the device, it is the same for all class objects (cuda or cpu is the same despite the class)
        state_batch = torch.tensor(state_batch, dtype=torch.float).to(self.actor.device)
        action_batch = torch.tensor(action_batch, dtype=torch.float).to(self.actor.device)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float).to(self.actor.device)
        new_state_batch = torch.tensor(new_state_batch, dtype=torch.float).to(self.actor.device)
        done_batch = torch.tensor(done_batch).to(self.actor.device)
        return state_batch, action_batch, reward_batch, new_state_batch, done_batch

    def save_models(self):
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()

    def learn(self):
        # deal with the situation in which we still not have filled the memory to batch size
        if self.memory.mem_counter < self.batch_size:
            return
        state_batch, action_batch, reward_batch, new_state_batch, done_batch = self.sample_transitions()

        ''' compute actor_target actions and critic_target values, then use obtained values to compute target y_i '''
        target_actions = self.actor_target(new_state_batch)#  + torch.tensor(self.noise(), dtype=torch.float).to(self.actor.device)
        target_critic_value_ = self.critic_target(new_state_batch, target_actions)
        # target_critic_value_next[done_batch==1] = 0.0  # if done_batch is integer valued
        target_critic_value_[done_batch] = 0.0  # if done_batch is bool -- see if it works this way
        target_critic_value_ = target_critic_value_.view(-1)  # necessary for operations on matching shapes
        target = reward_batch + self.gamma*target_critic_value_
        target = target.view(self.batch_size, 1)

        ''' zero out gradients '''
        self.actor.zero_grad()
        self.critic.zero_grad()
        ''' compute critic loss '''
        critic_value = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(target, critic_value)

        ''' compute actor loss'''
        # cannot directly use critic value, because it is evaluating a certain (s,a) pair.
        # The formula given in the paper - it appears that - wants to use critic to evaluate
        # the actions produced by an updated actor, given the state
        # actor_loss = torch.mean(critic_value)
        actor_loss = - self.critic(state_batch, self.actor(state_batch))
        actor_loss = torch.mean(actor_loss)

        critic_loss.backward()
        actor_loss.backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()

        self.update_network_parameters_phil()


    def __copy_params_phil(self, net_a, net_b, tau):
        net_a_params = net_a.named_parameters()
        net_b_params = net_b.named_parameters()
        net_a_state_dict = dict(net_a_params)
        net_b_state_dict = dict(net_b_params)
        for name in net_a_state_dict:
            net_a_state_dict[name] = tau*net_a_state_dict[name].clone() + (1 - tau) * net_b_state_dict[name].clone()
        return net_a_state_dict

    def update_network_parameters_phil(self, tau=None):
        if tau is None:
            tau = self.tau

        updated_actor_state_dict = self.__copy_params_phil(self.actor, self.actor_target, tau)
        updated_critic_state_dict = self.__copy_params_phil(self.critic, self.critic_target, tau)

        self.actor_target.load_state_dict(updated_actor_state_dict)
        self.critic_target.load_state_dict(updated_critic_state_dict)
