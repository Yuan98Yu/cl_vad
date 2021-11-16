import json


def save_args(cfg_path, args):
    with open(cfg_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def load_args(cfg_path, args):
    with open(cfg_path, 'r') as f:
        config = json.load(f)
        config.update(args.__dict__)
        args.__dict__ = config
