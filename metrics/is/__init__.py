from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader

from metrics.utils import is_valid_batch_size, is_valid_device_type
from model import load_model


class InceptionScore:
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

        self.model: nn.Module = load_model(self.model_type, False, self.device)

    def _get_predictions(self, x) -> np.ndarray:
        with torch.no_grad():
            x = self.model(x)
        predictions = softmax(x).data.cpu().numpy()
        return predictions

    @staticmethod
    def _get_mean_kl_divergence(
        x: np.ndarray, n_splits: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_samples: int = x.shape[0]
        n_split_samples: int = n_samples // n_splits

        scores = []
        for i in range(n_splits):
            _x = x[n_split_samples * i : n_split_samples * (i + 1), :]

            _x_mean = np.expand_dims(np.mean(_x, axis=0), axis=0)

            _kl = _x * (np.log(_x) - np.log(_x_mean))
            _kl = np.mean(np.sum(_kl, axis=1))
            scores.append(np.exp(_kl))

        return np.mean(scores), np.std(scores)

    def __call__(
        self, data_loader: DataLoader, n_feats: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_samples: int = len(data_loader)

        predictions = np.zeros((n_samples, n_feats), dtype=np.float32)
        for idx, data in enumerate(data_loader):
            _batch = self._get_predictions(data.to(self.device))
            predictions[
                self.batch_size * idx : self.batch_size * (idx + 1)
            ] = _batch

        mean, std = self._get_mean_kl_divergence(predictions, self.n_splits)
        return mean, std
