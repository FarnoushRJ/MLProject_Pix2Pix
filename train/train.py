import torch
from torch.utils.data import DataLoader
from barbar import Bar
from torch import mean, FloatTensor
from torch.optim import lr_scheduler
import matplotlib.pylab as plt
import numpy as np
import os

import sys
sys.path.insert(0, '../')
from datasets.general_dataset import GeneralDataset
from models.general_model import Pix2Pix


# ----------------------------------
def update_lr(epoch: int):
    """ Update the learning rate """
    lr_l = 1.0 - max(0, epoch + 1 - 100) / float(100 + 1)
    return lr_l


# ----------------------------------
def plot_loss(loss_G: list,
              loss_D: list,
              epochs: int):
    """
    :param loss_G: generator loss in each epoch
    :param loss_D: discriminator loss in each epoch
    :param epochs: number of training epochs
    """
    fig, ax = plt.subplots(dpi=100)
    ax.set_xlim(0, epochs)
    ax.set_ylim(0, max(np.max(loss_G), np.max(loss_D)) * 1.1)
    plt.plot(np.arange(epochs), loss_G, '.-r', label='Generator')
    plt.plot(np.arange(epochs), loss_D, '.-b', label='Discriminator')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss Values')
    plt.grid(True)
    plt.legend()
    plt.show()


# ----------------------------------
def train(opt):
    train_dataset = GeneralDataset(opt.dataset_name,
                                   opt.root,
                                   opt.train_folder,
                                   opt.no_crop,
                                   opt.no_flip,
                                   opt.crop_size,
                                   opt.scale)

    train_dataset = DataLoader(train_dataset,
                               batch_size=opt.batch_size,
                               shuffle=True)

    device = torch.cuda.current_device()
    torch.cuda.manual_seed(123)

    model = Pix2Pix(opt, True, device)
    model.to(device)
    model.train()

    schedulers = [lr_scheduler.LambdaLR(optimizer, lr_lambda=update_lr) for optimizer in
                  [model.optimizer_D, model.optimizer_G]]

    loss_D = list()
    loss_G = list()

    # training
    for epoch in range(opt.epochs):
        losses_D = list()
        losses_G = list()

        for data in Bar(train_dataset):

            model.set_input(data)  # set the input data
            model.optimize()  # calculate loss and update netD and netG
            losses_D.append(model.loss_D)
            losses_G.append(model.loss_G)

        avg_loss_D = mean(FloatTensor(losses_D))
        avg_loss_G = mean(FloatTensor(losses_G))

        print('------------------------------------')
        print("epoch: {} - loss_D: {} - loss_G: {}".format(epoch, avg_loss_D, avg_loss_G))

        loss_D.append(avg_loss_D)
        loss_G.append(avg_loss_G)

        # update the schedulers
        for scheduler in schedulers:
            scheduler.step()

    # save the models
    gen_path = os.path.join(opt.work_dir, '{}_generator_param.pkl'.format(opt.dataset_name))
    disc_path = os.path.join(opt.work_dir, '{}_discriminator_param.pkl'.format(opt.dataset_name))
    torch.save(model.netG.state_dict(), gen_path)
    torch.save(model.netD.state_dict(), disc_path)

    plot_loss(loss_G, loss_D, opt.epochs)

    return gen_path, disc_path


# ----------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '../')
    from arguments import arguments_parser
    parser = arguments_parser()
    opt = parser.parse_args()
    train(opt)
