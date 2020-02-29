from typing import Tuple

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST


def _get_mnist_transform(image_size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_mnist_dataset(dataset_path: str, image_size: int, mean: Tuple[float, float, float],
                      std: Tuple[float, float, float]) -> Dataset:
    return MNIST(
        root=dataset_path,
        transform=_get_mnist_transform(image_size, mean, std),
        download=True,
    )


def get_fashion_mnist_dataset(dataset_path: str, image_size: int, mean: Tuple[float, float, float],
                              std: Tuple[float, float, float]) -> Dataset:
    return FashionMNIST(
        root=dataset_path,
        transform=_get_mnist_transform(image_size, mean, std),
        download=True,
    )
