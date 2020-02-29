from enum import unique, Enum
from typing import Tuple

from torch.utils.data import Dataset, DataLoader

from data.celeba import get_celeba_dataset
from data.cifar import get_cifar10_dataset, get_cifar100_dataset
from data.custom import get_custom_dataset
from data.imagenet import get_imagenet_dataset
from data.lfw import get_lfw_dataset
from data.lsun import get_lsun_dataset
from data.mnist import get_mnist_dataset, get_fashion_mnist_dataset
from data.utils import get_normalize_type


@unique
class DataSetList(str, Enum):
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'
    MNIST = 'mnist'
    FASHION_MNIST = 'fashion_mnist'
    CELEBA = 'celeba'
    IMAGENET = 'imagenet'
    LSUN = 'lsun'
    LFW = 'lfw'
    CUSTOM = 'custom'


IMAGENET_NORMALIZE: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
)

NORMAL_NORMALIZE: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
)


def get_dataset(config, target: str) -> Dataset:
    image_size: int = int(config['data']['image_size'])
    image_crop_size: int = int(config['data']['crop_size'])
    dataset_type: str = config['data'][target]['dataset_type']
    mean, std = get_normalize_type(config)

    if dataset_type == DataSetList.CIFAR10:
        dataset = get_cifar10_dataset(dataset_type, image_size, mean, std)
    elif dataset_type == DataSetList.CIFAR100:
        dataset = get_cifar100_dataset(dataset_type, image_size, mean, std)
    elif dataset_type == DataSetList.MNIST:
        dataset = get_mnist_dataset(dataset_type, image_size, mean, std)
    elif dataset_type == DataSetList.FashionMNIST:
        dataset = get_fashion_mnist_dataset(dataset_type, image_size, mean, std)
    elif dataset_type == DataSetList.CELEBA:
        dataset = get_celeba_dataset(dataset_type, image_size, image_crop_size, mean, std)
    elif dataset_type == DataSetList.IMAGENET:
        dataset = get_imagenet_dataset(dataset_type, image_size, image_crop_size, mean, std)
    elif dataset_type == DataSetList.LSUN:
        dataset = get_lsun_dataset(dataset_type, config['data'][dataset_type]['classes'], image_size, mean, std)
    elif dataset_type == DataSetList.LFW:
        dataset = get_lfw_dataset(dataset_type, image_size, image_crop_size, mean, std)
    elif dataset_type == DataSetList.CUSTOM:
        dataset = get_custom_dataset(dataset_type, image_size, image_crop_size, mean, std)
    else:
        raise NotImplementedError(f'[-] not supported dataset : {dataset_type}')
    return dataset


def get_data_loader(config, target: str) -> DataLoader:
    dataset = get_dataset(config, target)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config['model']['batch_size'],
        shuffle=False,
        num_workers=config['n_threads'],
        drop_last=False,
    )
    return data_loader
