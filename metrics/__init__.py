from enum import Enum, unique

from metrics.utils import is_valid_batch_size, is_valid_device_type


@unique
class DeviceType(str, Enum):
    CPU = 'cpu'
    GPU = 'cuda'


@unique
class ModelType(str, Enum):
    VGG16 = 'vgg16'
    VGG19 = 'vgg19'
    RESNET18 = 'resnet_18'
    RESNET34 = 'resnet_34'
    RESNET50 = 'resnet_50'
    RESNET101 = 'resnet_101'
    RESNET152 = 'resnet_152'
    INCEPTION_V3 = 'inception_v3'


@unique
class MetricType(str, Enum):
    IS = 'inception_score'
    FID = 'fid'
    LPIPS = 'lpips'
    PPL = 'ppl'
    PSNR = 'psnr'
    SSIM = 'ssim'
    EMD = 'emd'
    MMD = 'mmd'
