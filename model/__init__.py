import torch.nn as nn
from torchvision.models.inception import inception_v3

from metrics import ModelType


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
