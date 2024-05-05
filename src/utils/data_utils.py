import json
import os



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
    

def format_data(data):
    formatted_data = []
    for round_num, round_info in data.items():
        formatted_round = {
            'turn'
        }