from enum import unique, Enum


@unique
class DataSetList(str, Enum):
    CIFAR10 = 'cifar10'
    CIFAR100 = 'cifar100'
    MNIST = 'mnist'
    CELEBA = 'celeba'
    IMAGENET = 'imagenet'
    LSUN = 'lsun'
    LFW = 'lfw'
    CUSTOM = 'custom'
