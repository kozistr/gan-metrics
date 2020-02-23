from enum import unique, Enum
from torch.utils.data import Dataset, DataLoader
from data.cifar import get_cifar10_dataset, get_cifar100_dataset
from data.mnist import get_mnist_dataset, get_fashion_mnist_dataset

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


def get_dataset(config) -> Dataset:
    dataset_type: str = config['data']['dataset_type']
    if dataset_type == DataSetList.CIFAR10:
        dataset = get_cifar10_dataset(config)
    elif dataset_type == DataSetList.CIFAR100:
        dataset = get_cifar100_dataset(config)
    elif dataset_type == DataSetList.MNIST:
        dataset = get_mnist_dataset(config)
    elif dataset_type == DataSetList.FashionMNIST:
        dataset = get_fashion_mnist_dataset(config)
    elif dataset_type == DataSetList.CELEBA:
        dataset = None
    elif dataset_type == DataSetList.IMAGENET:
        dataset = None
    elif dataset_type == DataSetList.LSUN:
        dataset = None
    elif dataset_type == DataSetList.LFW:
        dataset = None
    elif dataset_type == DataSetList.CUSTOM:
        dataset = None
    else:
        raise NotImplementedError(f'[-] not supported dataset : {dataset_type}')
    return dataset


def get_data_loader(config) -> DataLoader:
    dataset = get_dataset(config)
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['n_threads'],
        drop_last=False,
    )
    return data_loader
