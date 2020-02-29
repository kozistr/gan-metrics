from enum import Enum, unique


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
