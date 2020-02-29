from enum import unique, Enum
from typing import List

import torch.nn as nn
from torchvision.models.inception import inception_v3
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.vgg import vgg16, vgg19

from metrics.utils import is_valid_batch_size, is_valid_device_type


@unique
class DeviceType(str, Enum):
    CPU = 'cpu'
    GPU = 'cuda'


@unique
class ModelType(str, Enum):
    VGG16 = 'vgg16'
    VGG19 = 'vgg19'
    RESNET18 = 'resnet_18'
    RESNET34 = 'resnet_34'
    RESNET50 = 'resnet_50'
    RESNET101 = 'resnet_101'
    RESNET152 = 'resnet_152'
    INCEPTION_V3 = 'inception_v3'


IMAGENET_NORMALIZE_LIST: List[str] = [
    ModelType.VGG16, ModelType.VGG19, ModelType.RESNET18, ModelType.RESNET34, ModelType.RESNET50, ModelType.RESNET101,
    ModelType.RESNET152,
]
NORMAL_NORMALIZE_LIST: List[str] = [
    ModelType.INCEPTION_V3
]


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
    if model_type == ModelType.VGG16:
        model = vgg16(pretrained=True)
    elif model_type == ModelType.VGG19:
        model = vgg19(pretrained=True)
    elif model_type == ModelType.RESNET18:
        model = resnet18(pretrained=True)
    elif model_type == ModelType.RESNET34:
        model = resnet34(pretrained=True)
    elif model_type == ModelType.RESNET50:
        model = resnet50(pretrained=True)
    elif model_type == ModelType.RESNET101:
        model = resnet101(pretrained=True)
    elif model_type == ModelType.RESNET152:
        model = resnet152(pretrained=True)
    elif model_type == ModelType.INCEPTION_V3:
        model = inception_v3(pretrained=True, transform_input=False)
    else:
        raise NotImplementedError(f'[-] invalid model_type : {model_type}')

    if use_pooling_layer:
        model.fc = nn.Sequential()

    model.to(device)
    model.eval()

    return model
