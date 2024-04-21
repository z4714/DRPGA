import json




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