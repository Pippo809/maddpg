import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from make_env import make_env

class MultiAgentReplayBuffer:
    def __init__(self, max_size, state_dims, actor_dims,
                 n_actions, n_agents, batch_size):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.actor_dims = actor_dims
        self.n_actions = n_actions
        
        self.state_memory = np.zeros((self.mem_size, state_dims))
        self.new_state_memory = np.zeros((self.mem_size, state_dims))
        self.reward_memory = np.zeros((self.mem_size, self.n_agents))
        self.terminal_memory = np.zeros((self.mem_size, self.n_agents), dtype=bool)
        
        self.init_actor_memory()
    
    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []
        
        for i in range(self.n_agents):
            self.actor_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_new_state_memory.append(np.zeros((self.mem_size, self.actor_dims[i])))
            self.actor_action_memory.append(np.zeros((self.mem_size, self.n_actions)))
    
    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
                
        # Iterate over all the agents and store the observation in the specific agent 
        # that observed it
        for i in range(self.n_agents):
            self.actor_state_memory[i][index] = raw_obs[i]
            self.actor_new_state_memory[i][index] = raw_obs_[i]
            self.actor_action_memory[i][index] = action[i]
        
        self.mem_cntr += 1
    
    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        states = self.state_memory[batch]  # Flattened combination of all the observations
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        
        actor_states = []
        actor_new_states = []
        actions = []
        
        for agt_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agt_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agt_idx][batch])
            actions.append(self.actor_action_memory[agt_idx][batch])
        
        return actor_states, states, actions, rewards, actor_new_states, states_, terminal
    
    def is_batch_filled(self):
        if self.mem_cntr >= self.batch_size:
            return True
        return False