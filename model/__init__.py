import torch.nn as nn
from torchvision.models.inception import inception_v3

from metrics import ModelType


def load_model(model_type: str, device: str = 'cpu') -> nn.Module:
    if model_type == ModelType.INCEPTION_V3:
        model = inception_v3(pretrained=True, transform_input=False)
    elif model_type == ModelType.INCEPTION_V2:
        raise NotImplementedError(
            f'[-] not implemented model_type : {model_type}'
        )
    else:
        raise ValueError(f'[-] invalid model_type : {model_type}')

    model.to(device)
    model.eval()

    return model
