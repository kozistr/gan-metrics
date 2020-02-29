from typing import Tuple
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import LSUN


def _get_lsun_transform(image_size: int, mean: Tuple[float, float, float],
                        std: Tuple[float, float, float]):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_lsun_dataset(dataset_path: str, classes: str, image_size: int, mean: Tuple[float, float, float],
                     std: Tuple[float, float, float]) -> Dataset:
    return LSUN(
        root=dataset_path,
        transform=_get_lsun_transform(image_size, mean, std),
        classes=classes,
    )
