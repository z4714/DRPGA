from pettingzoo.mpe import simple_adversary_v3
import json
import numpy as np
import os
import torch
from ..src.maddpg_utils import SFSA_generator
from ..models.maddpg.sa_maddpg import MADDPG
from ..src.utils import rl_tools, model_utils


directory = "././data/SFSA_0200"

max_cycle = 200
seed = -1

test_env = simple_adversary_v3.parallel_env(max_cycles=max_cycle)


test_agents = ['adversary_0', 'agent_0', 'agent_1']

dialogues_and_order = {
    "adversary_0(Thief)": {"dialogues": [[] for _ in range(4)]},
    "agent_0(Elf Wizard)": {"dialogues": [[] for _ in range(4)]},
    "agent_1(Human Warrior)": {"dialogues": [[] for _ in range(4)]},
    "order": [[] for _ in range(4)]
}


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

state_dims=[]
action_dims=[]

for action_space_key, action_space in test_env.action_spaces.items():
    action_dims.append(action_space.n)
for state_space_key, state_space in test_env.observation_spaces.items():
    state_dims.append(state_space.shape[0])


critic_input_dim = sum(state_dims) +sum(action_dims)


sa_maddpg = MADDPG(test_env, device, actor_lr, critic_lr, hidden_dim, state_dims, action_dims, critic_input_dim, gamma, tau)
sa_maddpg = sa_maddpg.to(device)



params_date = ''
maddpg_dir = f'././parameters/weights/sa_maddpg/{params_date}'

actor_params = []
critic_params = []

actor_params = model_utils.load_params(actor_params, f'{maddpg_dir}/actor')
critic_params = model_utils.load_params(critic_params, f'{maddpg_dir}/critic')


for i,ddpgs in enumerate(sa_maddpg.agents):
    ddpgs.actor.load_state_dict(actor_params[i])
    ddpgs.critic.load_state_dict(critic_params[i])

    ddpgs.actor.eval()
    ddpgs.critic.eval()




print(sum(p.numel() for p in sa_maddpg.parameters())/1e6, 'M parameters in sa_maddpg')

times = 1000
for t in times:
    SFSA_generator.sfsa_generator(test_env, test_agents, seed, sa_maddpg, max_cycle, directory, dialogues_and_order)


