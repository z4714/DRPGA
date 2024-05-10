import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from copy import deepcopy
import collections
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from src.utils import  rl_tools
from models.utils import tk_func
from src.utils.model_utils import pad_list_to_length

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
        self.encoder = deepcopy(encoder)
        self.tokenzer = tokenizer
        self.tokenizer_fn = tokenizer_fn
        self.max_length = max_length
        self.obs_shape = state_dim
        self.action_shape = action_dim
        
        self.encoder_reduction = TwoLayerFC(en_out_dim, redu_dim, int(en_out_dim/2))
        self.actor = TwoLayerFC(state_dim+redu_dim, action_dim, hidden_dim)
        self.target_actor = TwoLayerFC(state_dim+redu_dim, action_dim,hidden_dim)
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
        self.device = device

    def combined_input(self, state, desc):
        desc_token = self.tokenzer(desc, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = desc_token['input_ids'].to(self.device)
        attention_mask = desc_token['attention_mask'].to(self.device)
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = encoder_outputs.last_hidden_state[:,0,:]

        reduced_features = self.encoder_reduction(last_hidden_state).squeeze(0)
        #print(state.shape)
        #print(reduced_features.shape)
        return torch.cat([state, reduced_features])   
    
    def take_action(self,state,desc, env_action_dim, explore=False):
        #print("state:",state)
        
        action = self.actor(self.combined_input(state,desc))[:env_action_dim]
        #print("action:",action)
        if explore:
            action = rl_tools.gumbel_softmax(action.unsqueeze(0))
        else: 
            action = rl_tools.onehot_from_logits(action.unsqueeze(0))
        return action.detach().cpu().numpy()[0]
    

    
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
            
            "glm": AutoTokenizer.from_pretrained("THUDM/chatglm3-6b-128k", trust_remote_code=True),
        }

        self.tokenizer = tokenizer_dict[tokenizer]

        for i in range(max_agents):
            print("ENDDPG",en_out_dim, redu_dim, state_dims[i], action_dims[i], 
                                    critic_input_dim, hidden_dim, actor_lr, 
                                    critic_lr, device)
            self.agents.append(ENDDPG(encoder, 
                                    self.tokenizer, tokenizer_fn, max_length,
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
    
    def take_action(self, states, desc, env_action_dims,used_agents,explore):
        


        cur_states = []
        

        try:
            for key, value in states.items():
                cur_states.append(value)
        #print("states.items",states.items())
            
            states = [torch.tensor(cur_states[i], dtype=torch.float32, device=self.device) for i in range(used_agents)]
        except Exception as e:
            print("states:",states)
            print("cur_state:",cur_states)
            print(e)
            


  

        return [agent.take_action(state, desc, env_action_dims[i], explore) for (i, agent), state in zip(enumerate(self.agents), states)]

    def update(self, obs, act, rew, next_obs, done, i_agent, desc, env_action_dims):
        cur_agent = self.agents[i_agent]
        cur_agent.critic_optimizer.zero_grad()
        #print(next_obs[i_agent].shape)
        all_target_act = [rl_tools.onehot_from_logits(pi(self.combined_input(agent,_next_obs,desc))) 
                          for pi, _next_obs, agent in zip(self.target_policies, next_obs, self.agents)]
        
        all_target_act_padded = pad_list_to_length(all_target_act, 6, (16,50))
        next_obs_padded = pad_list_to_length(next_obs, 6, (16,34))

        #concatenated = [torch.cat((all_target_act_padded[i], next_obs_padded[i]), dim=1) for i in range(len(all_target_act_padded))]
        #target_critic_input = torch.cat(concatenated, dim=1)
        
        #print(len(*all_target_act))
        #print(len(*next_obs))
        target_critic_input = torch.cat((*next_obs_padded, *all_target_act_padded), dim=1)
        #print(target_critic_input.shape)
    
       
        target_critic_value = rew[i_agent].view(-1,1) + self.gamma * cur_agent.target_critic(target_critic_input).mul(1-done[i_agent].view(-1,1))

        obs_padded = pad_list_to_length(act, 6, (16,34))
        act_padded = pad_list_to_length(obs, 6, (16,50))


        critic_input = torch.cat((*obs_padded, *act_padded), dim=1)



        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())

        critic_loss.backward()
        cur_agent.critic_optimizer.step()
        
        cur_agent.actor_optimizer.zero_grad()
        
        cur_actor_out = cur_agent.actor(self.combined_input(self.agents[i_agent],obs[i_agent],desc))
        cur_act_vf_in = rl_tools.gumbel_softmax(cur_actor_out)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == i_agent:
                all_actor_acs.append(cur_act_vf_in)
                
            else:
                all_actor_acs.append(rl_tools.onehot_from_logits(pi(self.combined_input(self.agents[i],_obs,desc))))
                
        
        obs_padded = pad_list_to_length(obs, 6, (16,34))
        all_actor_acs_padded = pad_list_to_length(all_actor_acs, 6, (16,50))
        #print(obs_padded[0].shape,len(obs_padded))
        #[print(all_actor_acs_i.shape, len(all_actor_acs_padded)) for all_actor_acs_i in all_actor_acs_padded]
        vf_in = torch.cat((*obs_padded, *all_actor_acs_padded), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()

        
        actor_loss += (cur_actor_out**2).mean()*1e-3
 
        actor_loss.backward()
        cur_agent.actor_optimizer.step()
        #target_critic_value = target_critic_value.expand_as(critic_value)

        
    def update_all_targets(self):
        for agt in self.agents:
            agt.soft_update(agt.actor, agt.target_actor, self.tau)
            agt.soft_update(agt.critic, agt.target_critic, self.tau)
    
    def combined_input(self, agent, state, desc):
        desc_token = self.tokenizer(desc, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = desc_token['input_ids'].to(self.device)
        attention_mask = desc_token['attention_mask'].to(self.device)
        encoder_outputs = agent.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = encoder_outputs.last_hidden_state[:,0,:]

        reduced_features = agent.encoder_reduction(last_hidden_state).squeeze(0).repeat(16,1)
        #print(state.shape)
        #print(reduced_features.shape)
        return torch.cat([state, reduced_features],dim=1)   
                               
