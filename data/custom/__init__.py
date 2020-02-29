from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def _get_custom_transform(config):
    image_size: int = int(config['data']['image_size'])
    crop_size: int = int(config['data']['crop_size'])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_custom_dataset(config) -> Dataset:
    return ImageFolder(
        root=config['data']['dataset_path'],
        transform=_get_custom_transform(config),
    )
