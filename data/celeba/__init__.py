from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def _get_celeba_transform(config, center_position: int = 138):
    image_size: int = int(config['data']['image_size'])
    return transforms.Compose([
        transforms.CenterCrop(center_position),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_celeba_dataset(config) -> Dataset:
    return ImageFolder(
        root=config['data']['dataset_path'],
        transform=_get_celeba_transform(config),
    )
