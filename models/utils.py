from torch.nn import Conv2d, ConvTranspose2d, init
from torch import Tensor


def init_func(m: Tensor,
              mean: float = 0,
              std: float = 0.02):
    """
    :param m: a tensor
    :param mean: mean value
    :param std: standard deviation
    :return:
    """
    if isinstance(m, Conv2d) or isinstance(m, ConvTranspose2d):
        if hasattr(m, 'weight'):
            init.normal_(m.weight.data, mean, std)

        if hasattr(m, 'bias'):
            init.constant_(m.bias.data, 0.0)
