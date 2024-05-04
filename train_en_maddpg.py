import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import collections
from datetime import datetime


from pettingzoo.mpe import simple_adversary_v3
from src.utils import rl_tools
from models.maddpg.sa_maddpg import MADDPG
from models.utils import persistence


max_cycles = 200
seed = -1

num_episodes = 100
episode_length = 64 
buffer_size = 100000
hidden_dim = 64
actor_lr = 1e-2
critic_lr = 1e-2
gamma = 0.95
tau = 1e-2
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
update_interval = 32
minimal_size = 15
replay_buffer = rl_tools.ReplayBuffer(buffer_size)




env = simple_adversary_v3.parallel_env(max_cycles=max_cycles)
observations, infos = env.reset() if seed == -1 else env.reset(seed=seed)


state_dims=[]
action_dims=[]

print(env.action_spaces)
print(env.observation_spaces)

for action_space_key, action_space in env.action_spaces.items():
    action_dims.append(action_space.n)
for state_space_key, state_space in env.observation_spaces.items():
    state_dims.append(state_space.shape[0])

print(state_dims)
print(action_dims)

critic_input_dim = sum(state_dims) +sum(action_dims)

sa_maddpg = MADDPG(env, device, actor_lr, critic_lr, hidden_dim, state_dims, action_dims, critic_input_dim, gamma, tau)
sa_maddpg = sa_maddpg.to(device)
print(sum(p.numel() for p in sa_maddpg.parameters())/1e9,'B parameters in sa_maddpg')

return_list = [] 
total_step = 0


agents = ['adversary_0', 'agent_0', 'agent_1']



for i_episode in range(num_episodes):   
   
    state, info = env.reset() if seed == -1 else env.reset(seed=seed)

    for e_i in range(episode_length):   
       
        actions = sa_maddpg.take_action(state, explore=True)
     
        env_actions = {agent: np.argmax(action) for agent, action in zip(agents, actions)}

        next_state, reward, done, truncations, infos = env.step(env_actions)
   
        replay_buffer.add(state, actions, reward, next_state, done)
        state = next_state

        total_step += 1

        if replay_buffer.size() >= minimal_size and total_step % update_interval == 0:
            obs, act, rew, next_obs, done = replay_buffer.sample(batch_size)
           
            obs_list, act_list, rew_list, next_obs_list, done_list = [], [], [], [], []

            for i, agent in enumerate(['adversary_0', 'agent_0', 'agent_1']):
                obs_list.append(np.array([s[agent] for s in obs]))
                act_list.append(np.array([a[i] for a in act]))
                rew_list.append(np.array([r[agent] for r in rew]))
                next_obs_list.append(np.array([ns[agent] for ns in next_obs]))
                done_list.append(np.array([d[agent] for d in done]))

            obs_tensor = [torch.tensor(o, dtype=torch.float32).to(device) for o in obs_list]
            act_tensor = [torch.tensor(a, dtype=torch.float32).to(device) for a in act_list]
            rew_tensor = [torch.tensor(r, dtype=torch.float32).to(device) for r in rew_list]                 
            next_obs_tensor = [torch.tensor(n, dtype=torch.float32).to(device) for n in next_obs_list]
            done_tensor = [torch.tensor(d, dtype=torch.float32).to(device) for d in done_list]  # 将布尔类型转换为浮点数类型


            for a_i in range(len(env.agents)):
                sa_maddpg.update(obs_tensor,act_tensor,rew_tensor,next_obs_tensor,done_tensor, a_i)#(要素，agent编号) #感觉可以改成一次传参优化速度
            sa_maddpg.update_all_targets()


    ep_returns = rl_tools.evaluate(env, sa_maddpg, n_episode=100)
    return_list.append(ep_returns)
   
    print("actions:", env_actions)
    print(f"Episode: {i_episode+1}, {ep_returns}")
  
             
env.close()

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

return_array = np.array(return_list)

return_info = np.array([max_cycles,seed,num_episodes])

np.savez(f'././evaluation/results/sa_maddpg/{current_time}.npz',return_array=return_array, return_info=return_info) 



persistence.save_actor_critic(sa_maddpg, 'sa_maddpg')

