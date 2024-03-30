from pettingzoo.mpe import simple_adversary_v3
import json
import numpy as np
import os

def convert_ndarray(data):
    if isinstance(data, dict):
        return {key: convert_ndarray(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.integer):
        return int(data)
    else: 
        return data

def generate_filename(directory, base_filename):
    

    files = os.listdir(directory)
    relevant_files = [f for f in files if f.startswith(base_filename) and f.endswith('.json')]
    next_num = len(relevant_files) + 1
    new_filename = f"{base_filename}_{next_num:06d}"
    return  os.path.join(directory, new_filename)
                      



max_cycle=200
test_env = simple_adversary_v3.parallel_env(max_cycles=max_cycle)

test_agents = ['adversary_0', 'agent_0', 'agent_1']


test_state, info = test_env.reset()   #重置环境并开始新的游戏轮次并获取初始状态
    
#print("state_0:",test_state)
#print("info:", info)
    # ep_returns = np.zeros(len(env.agents))

        #print(state)
    
    
    #print(e_i)
record_dict = {}    
    
cur_state = test_state    
last_actions={"adversary_0": -1, "agent_0": -1, "agent_1": -1}    
for i in range(max_cycle):
    
    test_actions = maddpg.take_action(cur_state, explore=False)#注意修改softmax此处会变
    

    #print("actions_{i}:",test_actions)
    #print(actions)

    
    test_env_actions = {t_agent: np.argmax(t_action) for t_agent, t_action in zip(test_agents, test_actions)}
    if test_env_actions == last_actions and i!=max_cycle-1:
        continue

    
    t_next_state, t_reward, t_done, t_truncations, t_infos = test_env.step(test_env_actions)

    

    
    round_data = {
        f"state_{i}": cur_state,
        f"actions_{i}": test_env_actions,
        f"reward_{i}": t_reward,
        f"state_{i+1}": t_next_state
    }
    
    record_dict[str(i)] = round_data
    

    cur_state = t_next_state
    last_actions = test_env_actions
    if t_reward == {}:
        print('done:',t_done)
        print('truncations:',t_truncations)
        break
    #print("t_done:",t_done)


record_dict_cov = convert_ndarray(record_dict)


directory = './datasets/SFSA/SFSA_0200/'
base_filename =  'SFSA_0200'

filename = generate_filename(directory, base_filename)


with open(filename, 'w') as json_file:
    json.dump(record_dict_cov, json_file, indent=2)