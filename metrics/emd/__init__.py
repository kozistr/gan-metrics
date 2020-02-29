import numpy as np
from ot import emd, emd2

from model import ModelWrapper


class EMD(ModelWrapper):
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

        super().__init__(
            model_type=model_type,
            batch_size=batch_size,
            n_splits=n_splits,
            device=device,
        )

    @staticmethod
    def _get_emd(
        x: np.ndarray, use_emd_2: bool = True, use_sqrt: bool = False
    ) -> np.ndarray:
        if use_sqrt:
            x = np.sqrt(np.abs(x))

        if use_emd_2:
            x = emd2([], [], x)
        else:
            x = emd([], [], x)
        return x

    def __call__(self, x, use_sqrt: bool = True):
        pass
