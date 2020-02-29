import yaml
from typing import Tuple
from data import IMAGENET_NORMALIZE, NORMAL_NORMALIZE
from model import NORMAL_NORMALIZE_LIST


def get_config(fn: str):
    with open(fn, 'r', encoding='utf8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


def get_normalize_type(config) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Get normalize type depends on the model type.
    :param config: yaml config. the configuration file.
    :return: float values. mean, std values
    """
    model_type: str = config['model']['model_type']
    if model_type in NORMAL_NORMALIZE_LIST:
        return NORMAL_NORMALIZE
    return IMAGENET_NORMALIZE
