import torch.nn as nn
from torchvision.models.inception import inception_v3

from metrics import ModelType
from metrics.utils import is_valid_batch_size, is_valid_device_type


class ModelWrapper:
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

        self.model_type = model_type
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.device = device

        if not is_valid_device_type(self.device):
            raise ValueError(f'[-] invalid device type : {self.device}')

        if not is_valid_batch_size(self.batch_size):
            raise ValueError(f'[-] invalid batch_size : {self.batch_size}')

        self.model: nn.Module = load_model(self.model_type, False, self.device)


def load_model(model_type: str, use_pooling_layer: bool = False, device: str = 'cpu') -> nn.Module:
    """
    loading the model by following model_type parameter
    :param model_type: str. type of model, default Inception-v3.
    :param use_pooling_layer: bool. use right after average pooling layer.
    :param device: str. type of device.
    :return: nn.Module. model instance.
    """
    if model_type == ModelType.INCEPTION_V3:
        model = inception_v3(pretrained=True, transform_input=False)
        if use_pooling_layer:
            model.fc = nn.Sequential()
    elif model_type == ModelType.INCEPTION_V2:
        raise NotImplementedError(
            f'[-] not implemented model_type : {model_type}'
        )
    else:
        raise ValueError(f'[-] invalid model_type : {model_type}')

    model.to(device)
    model.eval()

    return model
