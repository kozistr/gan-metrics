from enum import Enum, unique


@unique
class DeviceType(str, Enum):
    CPU = 'cpu'
    GPU = 'cuda'


@unique
class ModelType(str, Enum):
    INCEPTION_V2 = 'inception_v2'
    INCEPTION_V3 = 'inception_v3'
