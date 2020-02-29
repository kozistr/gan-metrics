from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import LSUN


def _get_lsun_transform(config):
    image_size: int = int(config['data']['image_size'])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_lsun_dataset(config) -> Dataset:
    return LSUN(
        root=config['data']['dataset_path'],
        transform=_get_lsun_transform(config),
        classes=config['data']['lsun']['classes'],
    )
