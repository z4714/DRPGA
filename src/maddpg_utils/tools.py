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
                      
