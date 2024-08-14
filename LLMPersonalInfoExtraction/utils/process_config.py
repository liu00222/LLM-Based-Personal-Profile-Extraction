import json


def open_config(config_path):
    return open_json(config_path)


def open_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def print_config(config, sort=False, indents=4):
    if type(config) == str:
        print(json.dumps(json.loads(config), sort_keys=sort, indent=indents))
    elif type(config) == dict:
        print(json.dumps(config, sort_keys=sort, indent=indents))
    else:
        raise ValueError(f"ERROR: Unsupported config {config}")
