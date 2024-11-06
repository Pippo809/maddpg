import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from make_env import make_env
from agent import Agent
from memory_buffer import MultiAgentReplayBuffer

class MADDPG:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, scenario="simple", alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01, chkp_dir="tmp/maddpg/"):
        self.agents: list[Agent] = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkp_dir += scenario
        
        for idx_agent in range(n_agents):
            self.agents.append(Agent(actor_dims[idx_agent], critic_dims, n_actions, idx_agent, chkp_dir, alpha, beta, fc1, fc2))
        
    def save_checkpoint(self):
        print("------Saving Checkpoint--------")
        for agent in self.agents:
            agent.save_models()
    
    def load_checkpoint(self):
        print("-------Loading Checkpoints------")
        for agent in self.agents:
            agent.load_models()
    
    def choose_action(self, observations):
        actions = []
        for idx, agent in enumerate(self.agents):
            action = agent.choose_action(observations[idx])
            actions.append(action)
        return actions
    
    def learn(self, memory: MultiAgentReplayBuffer):
        if not memory.is_batch_filled():
            return
        actor_states, states, actions, rewards, actor_states_, states_, dones, = memory.sample_buffer() 
        
        device = self.agents[0].actor.device
        
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_states_[agent_idx], 
                                 dtype=T.float).to(device)
            new_pi = agent.target_actor.forward(new_states)
            all_agents_new_actions.append(new_pi)
    
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        # Cost Function:
        for idx, agent in enumerate(self.agents):
            with T.no_grad():
                critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
                target = rewards[:, idx] + (1-dones[:,0].int())*agent.gamma*critic_value_
            
            critic_value = agent.critic.forward(states, old_actions).flatten()        
            
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()
            
            mu_states = T.tensor(actor_states[idx], dtype=T.float).to(device)
            old_actions_clone = old_actions.clone()
            old_actions_clone[:, idx*self.n_actions:idx*self.n_actions+self.n_actions] = agent.actor.forward(mu_states)
            
            # Basice REINFORCE, I take the gradient of the critic (the scores) with respect to the actor
            # I have to do this because I want to regress for each agent separately
            actor_loss = agent.critic.forward(states, old_actions_clone).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            
        for agent in self.agents:
            agent.update_netword_parameters()

