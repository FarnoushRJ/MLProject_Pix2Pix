from torch.nn import Module, BCEWithLogitsLoss, L1Loss
from torch.optim import Adam
from torch import Tensor, cat, ones, zeros
from torch.autograd import Variable

import sys
sys.path.insert(0, '../')
from models.discriminator import PixelGAN, PatchGAN
from models.generator import Generator


class Pix2Pix(Module):
    """
    pix2pix model implementation
    """

    # --------------------------
    def __init__(self, opt, training, device):
        """
        :param opt: options for creating the generator and discriminator
        """
        super(Pix2Pix, self).__init__()

        self.opt = opt
        self.training = training
        self.device = device

        if self.training:
            self.netG = Generator(self.opt.netG_in, self.opt.netG_out)
            self.netG.init_weight()
            self.netG.to(self.device)
            if self.opt.netD_name == 'PixelGAN':
                self.netD = PixelGAN(self.opt.netD_in, self.opt.netD_out)
            elif opt.netD_name == 'PatchGAN':
                self.netD = PatchGAN(self.opt.netD_in,
                                     self.opt.netD_out,
                                     self.opt.netD_layers)
            self.netD.init_weight()
            self.netD.to(self.device)
        else:
            self.netG = Generator(self.opt.netG_in, self.opt.netG_out)
            self.netG.init_weight()
            self.netG.to(self.device)

        if self.training:
            self.lossGAN = BCEWithLogitsLoss().to(self.device)
            self.lossL1 = L1Loss().to(self.device)

            self.optimizer_G = Adam(self.netG.parameters(), lr=opt.lr,
                                    betas=(self.opt.beta1, self.opt.beta2))
            self.optimizer_D = Adam(self.netD.parameters(), lr=opt.lr,
                                    betas=(self.opt.beta1, self.opt.beta2))

    # --------------------------
    def set_input(self, x: Tensor):
        """
        set the value of real_A and real_B based on the direction option
        """
        image_size = x.size()[2]
        if self.opt.direction == 'BtoA':
            self.real_A = x[:, :, :, 0:image_size].to(self.device)
            self.real_B = x[:, :, :, image_size:].to(self.device)

        elif self.opt.direction == 'AtoB':
            self.real_B = x[:, :, :, 0:image_size].to(self.device)
            self.real_A = x[:, :, :, image_size:].to(self.device)

    # --------------------------
    def forward(self):
        self.fake_B = self.netG.forward(self.real_A)

    # --------------------------
    def backward_D(self):
        """
        Discriminator back-propagation
        """
        # fake
        fake_AB = cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.lossGAN(pred_fake,
                                        Variable(zeros(pred_fake.size()).to(self.device)))

        # real
        real_AB = cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.lossGAN(pred_real,
                                        Variable(ones(pred_fake.size()).to(self.device)))

        # total loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * self.opt.netD_lambda
        self.loss_D.backward()

    # --------------------------
    def backward_G(self):
        """
        Generator back-propagation
        """
        # GAN loss
        fake_AB = cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.lossGAN(pred_fake,
                                       Variable(ones(pred_fake.size()).to(self.device)))

        # L1 loss
        self.loss_G_L1 = self.lossL1(self.fake_B, self.real_B) * self.opt.l1_lambda

        # total loss = GAN loss + L1 loss
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    # --------------------------
    def optimize(self):
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    # --------------------------
    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


