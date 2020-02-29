from typing import Tuple

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100


def _get_cifar_transform(image_size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_cifar10_dataset(dataset_path: str, image_size: int, mean: Tuple[float, float, float],
                        std: Tuple[float, float, float]) -> Dataset:
    return CIFAR10(
        root=dataset_path,
        transform=_get_cifar_transform(image_size, mean, std),
        download=True,
    )


def get_cifar100_dataset(dataset_path: str, image_size: int, mean: Tuple[float, float, float],
                         std: Tuple[float, float, float]) -> Dataset:
    return CIFAR100(
        root=dataset_path,
        transform=_get_cifar_transform(image_size, mean, std),
        download=True,
    )
