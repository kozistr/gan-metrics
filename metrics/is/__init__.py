from typing import Optional

import torch.nn as nn
from torchvision.models.inception import inception_v3

from metrics import ModelType
from metrics.utils import is_valid_batch_size, is_valid_device_type


class InceptionScore(nn.Module):
    def __init__(
        self,
        model_type: str = 'inception_v3',
        batch_size: int = 32,
        use_resize: bool = False,
        n_splits: int = 1,
        device: str = 'cpu',
    ):
        super().__init__()

        self.model_type = model_type
        self.batch_size = batch_size
        self.use_resize = use_resize
        self.n_splits = n_splits
        self.device = device

        self.model: Optional[nn.Module] = None

        if not is_valid_device_type(self.device):
            raise ValueError(f'[-] invalid device type : {self.device}')

        if not is_valid_batch_size(self.batch_size):
            raise ValueError(f'[-] invalid batch_size : {self.batch_size}')

    def load_network(self):
        if self.model_type == ModelType.INCEPTION_V3:
            self.model = inception_v3(pretrained=True, transform_input=False)
        elif self.model_type == ModelType.INCEPTION_V2:
            raise NotImplementedError(
                f'[-] not implemented model_type : {self.model_type}'
            )
        else:
            raise ValueError(f'[-] invalid model_type : {self.model_type}')

        self.model.to(self.device)
        self.model.eval()

    def __forward__(self, x):
        pass
