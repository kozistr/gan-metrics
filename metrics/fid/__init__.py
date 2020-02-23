from typing import Tuple

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from metrics.utils import is_valid_batch_size, is_valid_device_type
from model import load_model


class FID:
    def __init__(
        self,
        model_type: str = 'inception_v3',
        batch_size: int = 32,
        n_splits: int = 10,
        device: str = 'cpu',
    ):
        """
        Calculating Inception Score.
        :param model_type: str. name of model type, default inception v3.
        :param batch_size: int. batch size for feeding into the model.
        :param n_splits: int. number of splits for calculating the entropy.
        :param device: str. device type for loading the model.
        """
        super().__init__()

        self.model_type = model_type
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.device = device

        if not is_valid_device_type(self.device):
            raise ValueError(f'[-] invalid device type : {self.device}')

        if not is_valid_batch_size(self.batch_size):
            raise ValueError(f'[-] invalid batch_size : {self.batch_size}')

        self.model: nn.Module = load_model(self.model_type, True, self.device)

    def _get_activation(
        self, data_loader: DataLoader, n_feats: int = 2048
    ) -> np.ndarray:
        n_samples: int = len(data_loader)

        predictions = np.zeros((n_samples, n_feats), dtype=np.float32)
        for idx, data in enumerate(data_loader):
            _batch = self.model(data.to(self.device))
            predictions[
                self.batch_size * idx : self.batch_size * (idx + 1)
            ] = _batch
        return predictions

    @staticmethod
    def _get_activation_statistics(data_loader: DataLoader):
        pass

    def __call__(
        self, data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass
