import yaml


def get_config(fn: str):
    with open(fn, 'r', encoding='utf8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config
