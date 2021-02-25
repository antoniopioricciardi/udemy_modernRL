import torch
import numpy as np
import torch.nn.functional as F
from td3_networks import *
from replaymemory import ReplayMemory


class Agent():
    def __init__(self, load_checkpoint, checkpoint_file, env, n_states, n_actions, update_actor_interval=2, warmup=1000,
                 mem_size=10**6, batch_size=100, n_hid1=400, n_hid2=300, lr_alpha=1e-3, lr_beta=1e-3, gamma=0.99,
                 tau=5e-3, noise_mean=0, noise_sigma=0.1):

        self.load_checkpoint = load_checkpoint
        self.checkpoint_file = checkpoint_file
        # needed for clamping in the learn function
        self.env = env
        self.max_action = env.action_space.high[0]
        self.low_action = env.action_space.low[0]

        self.n_actions = n_actions
        # to keep track of how often we call "learn" function, for the actor network
        self.learn_step_counter = 0
        # to handle countdown to the end of the warmup period, incremented every time we call an action
        self.time_step = 0
        self.update_actor_interval = update_actor_interval
        self.warmup = warmup
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_mean = noise_mean
        self.noise_sigma = noise_sigma

        self.actor = TD3ActorNetwork(n_states, n_actions, n_hid1, n_hid2, lr_alpha, checkpoint_file, name='actor')
        self.target_actor = TD3ActorNetwork(n_states, n_actions, n_hid1, n_hid2, lr_alpha, checkpoint_file, name='target_actor')

        self.critic_1 = TD3CriticNetwork(n_states, n_actions, n_hid1, n_hid2, lr_beta, checkpoint_file, name='critic_1')
        self.critic_2 = TD3CriticNetwork(n_states, n_actions, n_hid1, n_hid2, lr_beta, checkpoint_file, name='critic_2')
        self.target_critic_1 = TD3CriticNetwork(n_states, n_actions, n_hid1, n_hid2, lr_beta, checkpoint_file, name='target_critic_1')
        self.target_critic_2 = TD3CriticNetwork(n_states, n_actions, n_hid1, n_hid2, lr_beta, checkpoint_file, name='target_critic_2')

        self.memory = ReplayMemory(mem_size, n_states, n_actions)

        # perform an exact copy of the networks to the respective targets
        # self.update_network_parameters(tau=1)
        self.update_network_parameters(self.actor, self.target_actor, tau=1)
        self.update_network_parameters(self.critic_1, self.target_critic_1, tau=1)
        self.update_network_parameters(self.critic_2, self.target_critic_2, tau=1)

    def choose_action(self, obs):
        if self.time_step < self.warmup:
            self.time_step += 1
            action = torch.tensor(self.env.action_space.sample())
        else:
            obs = torch.tensor(obs, dtype=torch.float).to(self.actor.device)
            action = self.actor(obs)

            # exploratory noise, scaled wrt action scale (max_action)
            noise = torch.tensor(np.random.normal(self.noise_mean, self.noise_sigma * self.max_action, size=self.n_actions))
            action += noise
        action = torch.clamp(action, self.low_action, self.max_action)
        return action.cpu().detach().numpy()

    def choose_action_eval(self, obs):
        obs = torch.tensor(obs, dtype=torch.float).to(self.actor.device)
        action = self.actor(obs)
        action = torch.clamp(action, self.low_action, self.max_action)
        return action.cpu().detach().numpy()

    def store_transition(self, obs, action, reward, obs_, done):
        self.memory.store_transition((obs, action, reward, obs_, done))

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

    def __copy_param(self, net_param_1, net_param_2, tau):
        # a.copy_(b) reads content from b and copy it to a
        for par, target_par in zip(net_param_1, net_param_2):
            #with torch.no_grad():
            val_to_copy = tau * par.weight + (1 - tau) * target_par.weight
            target_par.weight.copy_(val_to_copy)
            if target_par.bias is not None:
                val_to_copy = tau * par.bias + (1 - tau) * target_par.bias
                target_par.bias.copy_(val_to_copy)

    def update_network_parameters(self, network, target_network, tau=None):
        for par, target_par in zip(network.parameters(), target_network.parameters()):
            target_par.data.copy_(tau * par.data + (1 - tau) * target_par.data)

        #
        # # TODO: Controlla equivalenza con metodo Phil
        # # during the class initialization we call this method with tau=1, to perform an exact copy of the nets to targets
        # # then when we call this without specifying tau, we use the field stored
        # if tau is None:
        #     tau = self.tau
        #
        # actor_params = self.actor.children()
        # target_actor_params = self.target_actor.children()
        # self.__copy_param(actor_params, target_actor_params, tau)
        #
        # critic_params1 = self.critic_1.children()
        # target_critic_1_params = self.target_critic_1.children()
        # self.__copy_param(critic_params1, target_critic_1_params, tau)
        #
        # critic_params2 = self.critic_2.children()
        # target_critic_2_params = self.target_critic_2.children()
        # self.__copy_param(critic_params2, target_critic_2_params, tau)

    def learn(self):
        self.learn_step_counter += 1

        # deal with the situation in which we still not have filled the memory to batch size
        if self.memory.mem_counter < self.batch_size:
            return
        state_batch, action_batch, reward_batch, new_state_batch, done_batch = self.sample_transitions()
        # +- 0.5 as per paper. To be tested if min and max actions are not equal (e.g. -2 and 1)
        noise = np.clip(np.random.normal(self.noise_mean, 0.2, size=self.n_actions), -0.5, 0.5)
        target_next_action = torch.clamp(self.target_actor(new_state_batch) + noise, self.low_action, self.max_action)

        target_q1_ = self.target_critic_1(new_state_batch, target_next_action)
        target_q2_ = self.target_critic_1(new_state_batch, target_next_action)
        target_q_ = torch.min(target_q1_, target_q2_)  # take the min q_vale for every element in the batch
        target_q_[done_batch] = 0.0
        target = target_q_.view(-1)  # probably not needed
        target = reward_batch + self.gamma * target#_q
        target = target.view(self.batch_size, 1)  # probably not needed

        q_val1 = self.critic_1(state_batch, action_batch)
        q_val2 = self.critic_1(state_batch, action_batch)

        critic_loss1 = F.mse_loss(q_val1, target)
        critic_loss2 = F.mse_loss(q_val2, target)
        critic_loss = critic_loss1 + critic_loss2

        self.critic_1.zero_grad()
        self.critic_2.zero_grad()
        critic_loss.backward()
        #critic_loss1.backward()
        #critic_loss2.backward()

        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        if self.learn_step_counter % self.update_actor_interval:
            action = self.actor(action_batch)
            # compute actor loss proportional to the estimated value from q1 given state, action pairs, where the action
            # is recomputed using the new policy
            actor_loss = -torch.mean(self.critic_1(state_batch, action))

            self.actor.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_network_parameters(self.actor, self.target_actor, self.tau)
            self.update_network_parameters(self.critic_1, self.target_critic_1, self.tau)
            self.update_network_parameters(self.critic_2, self.target_critic_2, self.tau)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()
