import numpy as np
import torch

from model import DeviceType


def is_valid_device_type(device_type: str) -> bool:
    if device_type in DeviceType:
        if not torch.cuda.is_available() and device_type == DeviceType.GPU:
            return False
        return True
    return False


def is_valid_batch_size(batch_size: int) -> bool:
    return batch_size > 0


def is_valid_shape(query: np.ndarray, reference: np.ndarray) -> bool:
    return query.shape == reference.shape
