from data.utils import get_config
from . import  CONFIG_FILENAME
from data import get_data_loader


def main():
    config = get_config(CONFIG_FILENAME)

    real_data_loader = get_data_loader(config)
    fake_data_loader = get_data_loader(config)


main()
