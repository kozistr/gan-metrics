from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from typing import Tuple


def _get_celeba_transform(image_size: int, crop_size: int, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
    return transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_celeba_dataset(dataset_path: str, image_size: int, crop_size: int, mean: Tuple[float, float, float],
                        std: Tuple[float, float, float]) -> Dataset:
    return ImageFolder(
        root=dataset_path,
        transform=_get_celeba_transform(image_size, crop_size, mean, std),
    )
