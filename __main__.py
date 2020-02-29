from data.utils import get_config
from . import CONFIG_FILENAME
from data import get_data_loader
from typing import List


def main():
    config = get_config(CONFIG_FILENAME)

    real_data_loader = get_data_loader(config, target='real')
    fake_data_loader = get_data_loader(config, target='fake')

    metrics: List[str] = config['metrics']['metric_type']


main()
