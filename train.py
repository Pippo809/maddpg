import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from make_env import make_env
import time
from agent import Agent
from memory_buffer import MultiAgentReplayBuffer
from maddpg import MADDPG
from utils import obs_list_to_state_vector

def train(evaluate = False, scenario = "simple"):
    
    env = make_env(scenario)
    n_agents = env.n
    actor_dims = []
    
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    critic_dims = sum(actor_dims) + n_agents * env.action_space[0].n
    state_dims = sum(actor_dims)
    
    n_actions = env.action_space[0].n
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions, scenario, fc1=64, fc2=64, alpha=0.01, beta=0.01, chkp_dir="tmp/maddpg/")
    memory = MultiAgentReplayBuffer(100000, state_dims, actor_dims, n_actions, n_agents, batch_size=1024)
    
    PRINT_INTERVAL = 500
    N_GAMES = 100000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    best_score = -np.inf
    
    if evaluate:
        maddpg_agents.load_checkpoint()
        
    for i in range(N_GAMES):
        obs = env.reset()
        score = 0
        done = [False for _ in range(n_agents)]
        episode_steps = 0
        while not any(done):
            if evaluate:
                env.render()
                time.sleep(0.05)
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, done, info = env.step(actions)
            
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)
            
            memory.store_transition(obs, state, actions, reward, obs_, state_, done)
            
            if total_steps % 100 == 0 and not evaluate:
                maddpg_agents.learn(memory)
            
            obs = obs_
            score += sum(reward)
            total_steps += 1
            episode_steps += 1
            if episode_steps >= MAX_STEPS:
                done = [True for _ in range(n_agents)]
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
            if total_steps % PRINT_INTERVAL == 0:
                print(f"Steps: {total_steps}; Avg Score: {avg_score}")

if __name__ == "__main__":
    import torch as T
    T.autograd.set_detect_anomaly(True)
    train(evaluate=True, scenario="simple_adversary")
