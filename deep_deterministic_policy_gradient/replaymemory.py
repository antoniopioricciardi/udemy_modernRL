import numpy as np


class ReplayMemory:
    def __init__(self, size, input_size, n_actions):
        self.size = size
        self.input_size = input_size
        self.n_actions = n_actions
        self.mem_counter = 0
        self.state_memory = np.zeros((self.size, input_size))  # float32 is sufficient
        self.action_memory = np.zeros((self.size, n_actions))  # minor problem with int32, so now stick with this
        self.new_state_memory = np.zeros((self.size, input_size))
        self.reward_memory = np.zeros(self.size)
        self.terminal_memory = np.zeros(self.size, dtype=bool)#, dtype=np.uint8)  # later will be used as a mask to set everything to 0 for terminal states

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store memories in the position of first unoccupied memory
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        index = self.mem_counter % self.size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_state
        self.terminal_memory[index] = done
        #if index == 0:
            #self.mem_counter = 0
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.size)
        # max index of max_mem and shape of batch_size
        batch = np.random.choice(max_mem, batch_size)  # , replace=False)  # with replace=False sampled items are removed

        state_batch = self.state_memory[batch]
        action_batch = self.action_memory[batch]
        reward_batch = self.reward_memory[batch]
        new_state_batch = self.new_state_memory[batch]
        done_batch = self.terminal_memory[batch]

        return state_batch, action_batch, reward_batch, new_state_batch, done_batch
