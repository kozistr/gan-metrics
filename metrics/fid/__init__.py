from typing import Tuple

import numpy as np
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader

from metrics.utils import is_valid_shape
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

    def _get_activation_statistics(
        self, data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Getting mean over samples & covariance matrix of the features
        :param data_loader: DataLoader. image data.
        :return: Tuple[np.ndarray, np.ndarray]. mu, sigma. the statistics.
        """
        predictions = self._get_activation(data_loader)
        mu = np.mean(predictions, axis=0)
        sigma = np.cov(predictions, rowvar=False)
        return mu, sigma

    @staticmethod
    def _get_fid(
        mu_fake: np.ndarray,
        sigma_fake: np.ndarray,
        mu_real: np.ndarray,
        sigma_real: np.ndarray,
        epsilon: float = 1e-6,
    ) -> np.ndarray:
        """
        Getting Frechet Inception Distance between two multi-variate gaussian.
        :param mu_fake: np.ndarray. query mu for generated samples.
        :param sigma_fake: np.ndarray. query sigma for generated samples.
        :param mu_real: np.ndarray.
            reference mu for pre-calculated on an representative data.
        :param sigma_real: np.ndarray.
            reference sigma for pre-calculated on an representative data.
        :param epsilon: float. epsilon.
        :return:
        """
        mu_fake = np.atleast_1d(mu_fake)
        mu_real = np.atleast_1d(mu_real)

        if not is_valid_shape(mu_fake, mu_real):
            raise AssertionError(
                f'[-] incorrect shape. '
                f'fake : {mu_fake.shape} '
                f'real : {mu_real.shape}'
            )

        sigma_fake = np.atleast_2d(sigma_fake)
        sigma_real = np.atleast_2d(sigma_real)

        if not is_valid_shape(sigma_fake, sigma_real):
            raise AssertionError(
                f'[-] incorrect shape. '
                f'query : {sigma_fake.shape} '
                f'reference : {sigma_real.shape}'
            )

        mu = np.square(mu_fake - mu_real).sum()
        sigma, _ = sqrtm(np.dot(sigma_fake, sigma_real), disp=False)

        if not np.isfinite(sigma).all():
            offset = np.eye(sigma_fake.shape[0]) * epsilon
            sigma = sqrtm(np.dot(sigma_fake + offset, sigma_real + offset))

        if np.iscomplexobj(sigma):
            if not np.allclose(np.diagonal(sigma).imag, 0, atol=1e-3):
                m = np.max(np.abs(sigma.imag))
                raise ValueError(f'[-] imaginary component : {m}')
            sigma = np.real(sigma)

        distribution = mu + np.trace(sigma_fake + sigma_real - 2 * sigma)
        return distribution

    def __call__(
        self, fake_data_loader: DataLoader, real_data_loader: DataLoader
    ) -> np.ndarray:
        """
        :param fake_data_loader: DataLoader. generated samples.
        :param real_data_loader: DataLoader. real samples.
        :return: np.ndarray. frechet inception distance.
        """
        mu_fake, sigma_fake = self._get_activation_statistics(fake_data_loader)
        mu_real, sigma_real = self._get_activation_statistics(real_data_loader)
        fid = self._get_fid(mu_fake, sigma_fake, mu_real, sigma_real)
        return fid
