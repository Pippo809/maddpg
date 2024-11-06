import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from make_env import make_env
from actor_critic import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, actor_dims, critic_dims, n_actions, agent_name, chptdir, alpha=0.01, beta=0.01, fc1=64, fc2=64, gamma=0.99, tau=0.01):
        self.gamma = gamma
        self.tau = tau
        self.actor_dims = actor_dims
        self.critic_dims = critic_dims
        self.n_actions = n_actions
        self.agent_name = str(agent_name)
        n_agents = 1
        
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, self.agent_name+'_actor', chptdir)
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_actions, n_agents, self.agent_name+'_critic', chptdir)
        
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions, self.agent_name+'_target_actor', chptdir)
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_actions, n_agents, self.agent_name+'_target_critic', chptdir)
        
    def update_netword_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        actor_params = self.actor.named_parameters()
        target_actor_params: os.Iterator[F.Tuple[np.str | nn.Parameter]] = self.target_actor.named_parameters()
        
        critic_params: os.Iterator[F.Tuple[np.str | nn.Parameter]] = self.critic.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        
        actor_state_dict = dict(actor_params)
        target_actor_state_dict = dict(target_actor_params)
        critic_state_dict = dict(critic_params)
        target_critic_state_dict = dict(target_critic_params)
        
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                (1-tau)*target_actor_state_dict[name].clone()
        
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def choose_action(self, observation):
        observation = np.array(observation, dtype=np.float32)
        state = T.tensor(observation, dtype=T.float).unsqueeze(0).to(self.actor.device)
        actions = self.actor.forward(state)
        
        noise = T.rand(self.n_actions).to(self.actor.device)
        actions = actions + noise

        return actions.detach().cpu().numpy()[0]
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
        
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()        
        