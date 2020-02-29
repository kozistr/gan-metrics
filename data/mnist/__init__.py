from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST


def _get_mnist_transform(config):
    image_size: int = int(config['data']['image_size'])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_mnist_dataset(config) -> Dataset:
    return MNIST(
        root=config['data']['dataset_path'],
        transform=_get_mnist_transform(config),
        download=True,
    )


def get_fashion_mnist_dataset(config) -> Dataset:
    return FashionMNIST(
        root=config['data']['dataset_path'],
        transform=_get_mnist_transform(config),
        download=True,
    )
