import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import collections
from src.utils.model_utils import pad_to_shape


def onehot_from_logits(logits, eps=0.01):
  
    argmax_acs = (logits == logits.max(-1, keepdim=True)[0]).float()
    rand_acs = torch.autograd.Variable( 
        torch.eye(logits.shape[-1])[ 
            [np.random.choice(range(logits.shape[-1]),
                              
             size = logits.shape[0])]],
            requires_grad=False).to(logits.device)
    if argmax_acs.shape != rand_acs.shape:
        print("logits:",logits.shape)
        print("arg:",argmax_acs.shape)
        print("rand:",rand_acs.shape)
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in enumerate(torch.rand(logits.shape[0]))])



def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    U = torch.autograd.Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U+eps)+eps)


def gumbel_softmax_sample(logits, temperature=1.0):
  
    y=logits+sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y/temperature, dim=-1)   


def gumbel_softmax(logits, temperature=1.0):
    
    try: 
        
        y = gumbel_softmax_sample(logits, temperature)
        y_hard = onehot_from_logits(y)

        y = (y_hard.to(logits.device)-y).detach() +y
        return y
    except Exception as e:
        print("Shapes of y:", y.shape)
        print("Shapes of y_hard:", y_hard.shape)
        print("Shapes of logits:", logits.shape)
        print(e)



def evaluate(env, maddpg, n_episode=10, episode_length=25):

    returns = np.zeros(len(maddpg.agents))

    for _ in range(n_episode):
        obs, info = env.reset()
        for t_i in range(episode_length):
            actions = maddpg.take_action(obs, explore=False)

            e_acts = {agent: np.argmax(action) for agent, action in zip(env.agents, actions)}
            obs, rew, done,trun, info = env.step(e_acts)
            rew_value = np.array(list(rew.values()),dtype=np.float64)
    
            returns += rew_value 
    returns /= n_episode   
    return returns.tolist()

def en_evaluate(env, maddpg, desc, env_action_dims,used_agents, n_episode=10, episode_length=25):

    returns = np.zeros(len(used_agents))

    for _ in range(n_episode):
        obs, info = env.reset()
        obs_padded = {
        agent_name: pad_to_shape(obs[agent_name], agent.obs_shape)
        for agent_name, agent in zip(obs.keys(), maddpg.agents)
        }
        for t_i in range(episode_length):
            actions = maddpg.take_action(obs_padded,desc,env_action_dims,used_agents, explore=False)

            e_acts = {agent: np.argmax(action) for agent, action in zip(env.agents, actions)}
            obs, rew, done,trun, info = env.step(e_acts)
            rew_value = np.array(list(rew.values()),dtype=np.float64)
            #print(rew_value)
            returns += np.mean(rew_value) 
    returns /= n_episode   
    return returns.tolist()

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen = capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done=zip(*transitions)
     
        return np.array(state), action, reward, np.array(next_state), done
    
    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size])/window_size
   
    r = np.arange(1, window_size-1, 2)
    
    begin = np.cumsum(a[:window_size-1])[::2]/r
    end = (np.cumsum(a[:-window_size:-1])[::2]/r)[::-1]

    return np.concatenate((begin,middle,end))