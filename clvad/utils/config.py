import json


def save_args(cfg_path, args):
    with open(cfg_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def load_args(cfg_path, args):
    with open(cfg_path, 'w') as f:
        args.__dict__ = json.load(f)
