from torch.nn import Module, Conv2d, LeakyReLU, BatchNorm2d, Sequential
from torch import Tensor

import sys
sys.path.insert(0, '../')
from models.utils import init_func


# -------------------------------
class PatchGAN(Module):
    """
    PatchGAN discriminator (default=70x70)
    """
    def __init__(self,
                 in_channels: int = 6,
                 out_channels: int = 64,
                 n_layers: int = 3):

        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param n_layers: 1, 3 or 5
        """
        super(PatchGAN, self).__init__()
        factor = 1
        # C64
        sequence = [
            Conv2d(in_channels, out_channels,
                   kernel_size=4, stride=2, padding=1)
        ]

        for i in range(0, n_layers):
            if i <= 2:
                if n_layers == 3 and i == 2:
                    stride = 1
                else:
                    stride = 2
                factor = 2 ** i
                sequence += [
                    Conv2d(out_channels*factor, out_channels*factor*2,
                           kernel_size=4, stride=stride, padding=1),
                    BatchNorm2d(out_channels*factor*2),
                    LeakyReLU(0.2, True)
                ]
            elif i >= 3:
                if n_layers == 5 and i == 4:
                    stride = 1
                else:
                    stride = 2
                factor = 8
                sequence += [
                    Conv2d(out_channels*factor, out_channels*factor,
                           kernel_size=4, stride=stride, padding=1),
                    BatchNorm2d(out_channels*factor),
                    LeakyReLU(0.2, True)
                ]

        if n_layers == 3 or n_layers == 1:
            factor = 2 * factor

        # one channel output
        sequence += [Conv2d(out_channels*factor, 1,
                            kernel_size=4, stride=1, padding=1)]

        self.model = Sequential(*sequence)

    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)

    def init_weight(self, mean: float = 0, std: float = 0.02):
        """
        :param mean: mean value
        :param std: standard deviation
        :return: void
        """
        for m in self._modules:
            init_func(self._modules[m], mean, std)


# -------------------------------
class PixelGAN(Module):
    """
    1x1 PatchGAN discriminator --> PixelGAN
    """
    def __init__(self,
                 in_channels: int = 6,
                 out_channels: int = 64):
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        """
        super(PixelGAN, self).__init__()

        # Layer1: C64
        self.conv1 = Conv2d(in_channels, out_channels,
                            kernel_size=1, stride=1, padding=0)
        self.relu1 = LeakyReLU(0.2, True)

        # Layer2: C128
        self.conv2 = Conv2d(out_channels, out_channels * 2,
                            kernel_size=1, stride=1, padding=0)
        self.conv2_norm = BatchNorm2d(out_channels * 2)
        self.relu2 = LeakyReLU(0.2, True)

        # Layer 3
        self.conv3 = Conv2d(out_channels * 2, 1,
                            kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor
        :return: output tensor
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2_norm(self.conv2(x)))
        x = self.conv3(x)
        return x

    def init_weight(self, mean: float = 0, std: float = 0.02):
        """
        :param mean: mean value
        :param std: standard deviation
        :return: void
        """
        for m in self._modules:
            init_func(self._modules[m], mean, std)

