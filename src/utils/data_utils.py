import json
import os
from collections import OrderedDict


def json2Tpath(data, path=""):
    if isinstance(data, dict):
        text = []
        for key, value in data.items():
            full_path = f"{path}.{key}" if path else key
            text.extend(json2Tpath(value, full_path))
        return text
    elif isinstance(data, list):
        text = []
        for idx, item in enumerate(data):
            full_path = f"{path}[{idx}]"
            text.extend(json2Tpath(item, full_path))
        return text
    elif isinstance(data, str):
        return [(path, data)]
    else:
        return []


def find_latest_file(dir):
    spec_file = ''
    pth_files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.pth')]
    latest_file = max(pth_files, key=os.path.getmtime)
    return latest_file

def round_nested(data):
    if isinstance(data, list):
        return [round_nested(x) for x in data]
    elif isinstance(data, dict):
        return {k: round_nested(v) for k, v in data.items()}
    elif isinstance(data, float):
        return round(data, 2)
    else:
        return data
    

def format_data_sfsa(data,agents):
    formatted_data = []
    for round_num, round_info in data.items():
        next_round = 'state_'+str(int(round_num) + 1)
        curr_round = 'state_'+round_num
        details = OrderedDict()
        details['state'] = {
                    agents[0]:round_nested(round_info[curr_round]['adversary_0']),
                    agents[1]:round_nested(round_info[curr_round]['agent_0']),
                    agents[2]:round_nested(round_info[curr_round]['agent_1']),
        
        },
        
        details['actions'] = round_info['actions_'+round_num],
        details['reward']= round_info['reward_'+round_num],

        details['next_state'] ={
                    agents[0]:round_nested(round_info[next_round]['adversary_0']),
                    agents[1]:round_nested(round_info[next_round]['agent_0']),
                    agents[2]:round_nested(round_info[next_round]['agent_1']),
        }
                    
                
        formatted_round = {
            'turn': 'round' + round_num,
            'details': details,
          
 
        }
        formatted_data.append(formatted_round)
    return formatted_data