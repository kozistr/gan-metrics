from typing import Tuple

import numpy as np
from torch.utils.data import DataLoader

from model import ModelWrapper


class FID(ModelWrapper):
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
        super().__init__(
            model_type=model_type,
            batch_size=batch_size,
            n_splits=n_splits,
            device=device,
        )

    def _get_activation(self):
        pass

    @staticmethod
    def _get_activation_statistics(data_loader: DataLoader):
        pass

    def __call__(
        self, data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass
