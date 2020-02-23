from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100


def _get_cifar_transform(config):
    image_size: int = int(config['data']['image_size'])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_cifar10_dataset(config) -> Dataset:
    return CIFAR10(
        root=config['data']['dataset_path'],
        transform=_get_cifar_transform(config),
        download=True,
    )


def get_cifar100_dataset(config) -> Dataset:
    return CIFAR100(
        root=config['data']['dataset_path'],
        transform=_get_cifar_transform(config),
        download=True,
    )
