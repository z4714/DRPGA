import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import collections
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from src.utils import  rl_tools
from models.utils import tk_func

class TwoLayerFC(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ENDDPG(torch.nn.Module):
    def __init__(self, encoder, 
                 tokenizer, tokenizer_fn, max_length,
                 en_out_dim, redu_dim, state_dim, action_dim, critic_input_dim, 
                 hidden_dim, actor_lr, critic_lr, device):
        super().__init__()
        self.encoder = encoder
        self.tokenzer = tokenizer
        self.tokenizer_fn = tokenizer_fn
        self.max_length = max_length
        self.encoder_reduction = TwoLayerFC(en_out_dim, redu_dim, en_out_dim/2)
        self.actor = TwoLayerFC(state_dim, action_dim, hidden_dim)
        self.target_actor = TwoLayerFC(state_dim, action_dim,hidden_dim)
        self.critic = TwoLayerFC(critic_input_dim, 1, hidden_dim)
        self.target_critic = TwoLayerFC(critic_input_dim, 1, hidden_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=actor_lr)
        
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(),lr=critic_lr)
        self.tokenizer_fn_dict = {
            "tokenizer_fn": tk_func.tokenizer_fn,
            "collate_fn": tk_func.collate_fn,
        }
        
    def take_action(self,state, explore=False):
        #print("state:",state)
        action = self.actor(state)
        #print("action:",action)
        if explore:
            action = rl_tools.gumbel_softmax(action.unsqueeze(0))
        else: 
            action = rl_tools.onehot_from_logits(action.unsqueeze(0))
        return action.detach().cpu().numpy()[0]
    
    def actor_forward(self, state, desc):
        desc = self.tokenzer(desc, max_length=self.max_length,padding="max_length", truncation=True, return_tensors="pt")
        encoder_output = self.encoder(**desc)


    def critic_forward(self, state, action):
        return self.critic(torch.cat((state, action), dim=1))
    
    
    def soft_update(self, net, target_net, tau):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data*(1.0-tau)+param.data*tau)


class ENMADDPG(torch.nn.Module):
    def __init__(self, tokenizer, 
                 tokenizer_fn, max_length, 
                 encoder, en_out_dim, 
                 redu_dim, max_agents, 
                 device, actor_lr, critic_lr, 
                 hidden_dim, state_dims, action_dims, 
                 critic_input_dim, gamma, tau):
        super().__init__()
        self.agents = torch.nn.ModuleList()
        tokenizer_dict = {
            "bert-base-uncased": BertTokenizer.from_pretrained("bert-base-uncased"),
            "microsoft/codebert-base": AutoTokenizer.from_pretrained("microsoft/codebert-base"),
            "gpt2": GPT2Tokenizer.from_pretrained("gpt2"),
            "spt": '',
            "glm": AutoTokenizer.from_pretrained("THUDM/chatglm3-6b-128k", trust_remote_code=True),
        }

        tokenizer = tokenizer_dict[tokenizer]

        for i in range(max_agents):
            print("ENDDPG",en_out_dim, redu_dim, state_dims[i], action_dims[i], 
                                    critic_input_dim, hidden_dim, actor_lr, 
                                    critic_lr, device)
            self.agents.append(ENDDPG(encoder, 
                                    tokenizer, tokenizer_fn, max_length,
                                    en_out_dim, redu_dim, 
                                    state_dims[i], action_dims[i], 
                                    critic_input_dim, hidden_dim, 
                                    actor_lr, critic_lr, device)).to(device)
        self.gamma = gamma
        self.tau = tau
        self.critic_criterion = torch.nn.MSELoss()
        self.device = device
        
    @property 
    def policies(self):
        return [agt.actor for agt in self.agents]
    
    @property  
    def target_policies(self):
        return [agt.target_actor for agt in self.agents]
    
    def take_action(self, states, explore):
        
        cur_states = []
        

        try:
            for key, value in states.items():
                cur_states.append(value)
        #print("states.items",states.items())
            states = [torch.tensor(cur_states[i], dtype=torch.float32, device=self.device) for i, agent in enumerate(self.agents)]
        except Exception as e:
            print("states:",states)
            print("cur_state:",cur_states)
            print(e)
            




        return [agent.take_action(state, explore) for agent, state in zip(self.agents, states)]

    def update(self, obs, act, rew, next_obs, done, i_agent):
        cur_agent = self.agents[i_agent]
        cur_agent.critic_optimizer.zero_grad()
        all_target_act = [rl_tools.onehot_from_logits(pi(_next_obs)) 
                          for pi, _next_obs in zip(self.target_policies, next_obs)]
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        
    
       
        target_critic_value = rew[i_agent].view(-1,1) + self.gamma * cur_agent.target_critic(target_critic_input).mul(1-done[i_agent].view(-1,1))

     
        critic_input = torch.cat((*obs, *act), dim=1)



        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())

        critic_loss.backward()
        cur_agent.critic_optimizer.step()
        
        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[i_agent])
        cur_act_vf_in = rl_tools.gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
                
            else:
                all_actor_acs.append(rl_tools.onehot_from_logits(pi(_obs)))
                
 
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()

        
        actor_loss += (cur_actor_out**2).mean()*1e-3
 
        actor_loss.backward()
        cur_agent.actor_optimizer.step()
        #target_critic_value = target_critic_value.expand_as(critic_value)

        
    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)
                               