import numpy as np
import torch.nn as nn
from ot import emd2

from metrics.utils import is_valid_batch_size, is_valid_device_type
from model import load_model


class EMD:
    def __init__(
        self,
        model_type: str = 'inception_v3',
        batch_size: int = 32,
        n_splits: int = 10,
        device: str = 'cpu',
    ):
        """
        Calculating Earth Mover's Distance.
        :param model_type: str. name of model type, default inception v3.
        :param batch_size: int. batch size for feeding into the model.
        :param n_splits: int. number of splits for calculating the entropy.
        :param device: str. device type for loading the model.
        """

        self.model_type = model_type
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.device = device

        if not is_valid_device_type(self.device):
            raise ValueError(f'[-] invalid device type : {self.device}')

        if not is_valid_batch_size(self.batch_size):
            raise ValueError(f'[-] invalid batch_size : {self.batch_size}')

        self.model: nn.Module = load_model(self.model_type, True, self.device)

    @staticmethod
    def _get_emd(x: np.ndarray, use_sqrt: bool = False) -> np.ndarray:
        if use_sqrt:
            x = np.sqrt(np.abs(x))
        x = emd2([], [], x)
        return x

    def __call__(self, x, use_sqrt: bool = True):
        pass
