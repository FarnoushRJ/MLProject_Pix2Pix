from torch.nn import Module, Conv2d, BatchNorm2d, LeakyReLU, ConvTranspose2d, ReLU, Tanh, Dropout
from torch import Tensor, cat

import sys
sys.path.insert(0, '../')
from models.utils import init_func


class Generator(Module):
    """
    UNet Generator Architecture
    """
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 64
                 ):
        """
        :param in_channels: number of input channels [2 or 3]
        :param out_channels: number of output channels
        """
        super(Generator, self).__init__()

        # UNet Encoder --> downsample by a factor of 2
        # Layer1: C64
        self.conv1 = Conv2d(in_channels,
                            out_channels,
                            kernel_size=4, stride=2, padding=1)

        # Layer2: C128
        self.leaky_relu2 = LeakyReLU(0.2, True)
        self.conv2 = Conv2d(out_channels,
                            out_channels * 2,
                            kernel_size=4, stride=2, padding=1)
        self.conv2_norm = BatchNorm2d(out_channels * 2)

        # Layer3: C256
        self.leaky_relu3 = LeakyReLU(0.2, True)
        self.conv3 = Conv2d(out_channels * 2,
                            out_channels * 4,
                            kernel_size=4, stride=2, padding=1)
        self.conv3_norm = BatchNorm2d(out_channels * 4)

        # Layer4: C512
        self.leaky_relu4 = LeakyReLU(0.2, True)
        self.conv4 = Conv2d(out_channels * 4,
                            out_channels * 8,
                            kernel_size=4, stride=2, padding=1)
        self.conv4_norm = BatchNorm2d(out_channels * 8)

        # Layer5: C512
        self.leaky_relu5 = LeakyReLU(0.2, True)
        self.conv5 = Conv2d(out_channels * 8,
                            out_channels * 8,
                            kernel_size=4, stride=2, padding=1)
        self.conv5_norm = BatchNorm2d(out_channels * 8)

        # Layer6: C512
        self.leaky_relu6 = LeakyReLU(0.2, True)
        self.conv6 = Conv2d(out_channels * 8,
                            out_channels * 8,
                            kernel_size=4, stride=2, padding=1)
        self.conv6_norm = BatchNorm2d(out_channels * 8)

        # Layer7: C512
        self.leaky_relu7 = LeakyReLU(0.2, True)
        self.conv7 = Conv2d(out_channels * 8,
                            out_channels * 8,
                            kernel_size=4, stride=2, padding=1)
        self.conv7_norm = BatchNorm2d(out_channels * 8)

        # Layer8: C512
        self.leaky_relu8 = LeakyReLU(0.2, True)
        self.conv8 = Conv2d(out_channels * 8,
                            out_channels * 8,
                            kernel_size=4, stride=2, padding=1)

        # UNet Decoder --> upsample by a factor of 2
        # Layer1: CD512
        self.dropout1 = Dropout(0.5)
        self.relu1 = ReLU(0.2)
        self.deconv1 = ConvTranspose2d(out_channels * 8,
                                       out_channels * 8,
                                       kernel_size=4, stride=2, padding=1)
        self.deconv1_norm = BatchNorm2d(out_channels * 8)

        # Layer2: CD512
        self.dropout2 = Dropout(0.5)
        self.relu2 = ReLU(0.2)
        self.deconv2 = ConvTranspose2d(out_channels * 8 * 2,
                                       out_channels * 8,
                                       kernel_size=4, stride=2, padding=1)
        self.deconv2_norm = BatchNorm2d(out_channels * 8)

        # Layer3: CD512
        self.dropout3 = Dropout(0.5)
        self.relu3 = ReLU(0.2)
        self.deconv3 = ConvTranspose2d(out_channels * 8 * 2,
                                       out_channels * 8,
                                       kernel_size=4, stride=2, padding=1)
        self.deconv3_norm = BatchNorm2d(out_channels * 8)

        # Layer4: C512
        self.relu4 = ReLU(0.2)
        self.deconv4 = ConvTranspose2d(out_channels * 8 * 2,
                                       out_channels * 8,
                                       kernel_size=4, stride=2, padding=1)
        self.deconv4_norm = BatchNorm2d(out_channels * 8)

        # Layer5: C256
        self.relu5 = ReLU(0.2)
        self.deconv5 = ConvTranspose2d(out_channels * 8 * 2,
                                       out_channels * 4,
                                       kernel_size=4, stride=2, padding=1)
        self.deconv5_norm = BatchNorm2d(out_channels * 4)

        # Layer6: C128
        self.relu6 = ReLU(0.2)
        self.deconv6 = ConvTranspose2d(out_channels * 4 * 2,
                                       out_channels * 2,
                                       kernel_size=4, stride=2, padding=1)

        self.deconv6_norm = BatchNorm2d(out_channels * 2)

        # Layer7: C64
        self.relu7 = ReLU(0.2)
        self.deconv7 = ConvTranspose2d(out_channels * 2 * 2,
                                       out_channels,
                                       kernel_size=4, stride=2, padding=1)
        self.deconv7_norm = BatchNorm2d(out_channels)

        # Layer8
        self.relu8 = ReLU(0.2)
        self.deconv8 = ConvTranspose2d(out_channels * 2,
                                       in_channels,
                                       kernel_size=4, stride=2, padding=1)
        self.tanh = Tanh()

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor
        :return: output tensor
        """
        # UNet Encoder
        x1 = self.conv1(x)
        x2 = self.conv2_norm(self.conv2(self.leaky_relu2(x1)))
        x3 = self.conv3_norm(self.conv3(self.leaky_relu3(x2)))
        x4 = self.conv4_norm(self.conv4(self.leaky_relu4(x3)))
        x5 = self.conv5_norm(self.conv5(self.leaky_relu5(x4)))
        x6 = self.conv6_norm(self.conv6(self.leaky_relu6(x5)))
        x7 = self.conv7_norm(self.conv7(self.leaky_relu7(x6)))
        x8 = self.conv8(self.leaky_relu8(x7))

        # UNet Decoder
        y1 = self.dropout1(self.deconv1_norm(self.deconv1(self.relu1(x8))))
        y1 = cat([y1, x7], 1)
        y2 = self.dropout2(self.deconv2_norm(self.deconv2(self.relu2(y1))))
        y2 = cat([y2, x6], 1)
        y3 = self.dropout3(self.deconv3_norm(self.deconv3(self.relu1(y2))))
        y3 = cat([y3, x5], 1)
        y4 = self.deconv4_norm(self.deconv4(self.relu4(y3)))
        y4 = cat([y4, x4], 1)
        y5 = self.deconv5_norm(self.deconv5(self.relu5(y4)))
        y5 = cat([y5, x3], 1)
        y6 = self.deconv6_norm(self.deconv6(self.relu6(y5)))
        y6 = cat([y6, x2], 1)
        y7 = self.deconv7_norm(self.deconv7(self.relu7(y6)))
        y7 = cat([y7, x1], 1)
        y8 = self.deconv8(self.relu8(y7))
        out = self.tanh(y8)

        return out

    def init_weight(self, mean: float = 0, std: float = 0.02):
        """
        :param mean: mean value
        :param std: standard deviation
        :return: void
        """
        for m in self._modules:
            init_func(self._modules[m], mean, std)
