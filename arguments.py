import argparse

def arguments_parser():

    parser = argparse.ArgumentParser()
    
    # --------------------------
    # Discriminator
    parser.add_argument('--netD_name', required=False, help='PixelGAN | PatchGAN', default='PatchGAN')
    parser.add_argument('--netD_layers', required=False, help='1 | 3 | 5', default=3, type=int)
    parser.add_argument('--netD_in', required=False, help='number of input channels', default=6, type=int)
    parser.add_argument('--netD_out', required=False, help='number of output channels', default=64, type=int)
    parser.add_argument('--netD_lambda', required=False, help='lambda for discriminator total loss', default=0.5, type=float)
    
    # --------------------------
    # Generator
    parser.add_argument('--netG_in', required=False, help='number of input channels', default=3, type=int)
    parser.add_argument('--netG_out', required=False, help='number of output channels', default=64, type=int)
    parser.add_argument('--l1_lambda', required=False, help='lambda for L1 loss', default=100.0, type=float)
    
    # --------------------------
    # Training
    parser.add_argument('--lr', required=False, help='learning rate', default=0.0002, type=float)
    parser.add_argument('--beta1', required=False, help='beta1', default=0.5, type=float)
    parser.add_argument('--beta2', required=False, help='beta2', default=0.999, type=float)
    parser.add_argument('--batch_size', required=False, help='input batch size', default=1, type=int)
    parser.add_argument('--epochs', required=False, help='number of epochs', default=200, type=int)  
    parser.add_argument('--work_dir', required=False, help='working directory', default=".")
    
    # --------------------------
    # Dataset
    parser.add_argument('--dataset_name', required=False, help='facades | maps', default='facades')
    parser.add_argument('--direction', required=False, help='AtoB | BtoA', default='AtoB')
    parser.add_argument('--root', required=True, help='path to the dataset')
    parser.add_argument('--train_folder', required=False, help='train | test | val', default='train')
    parser.add_argument('--test_folder', required=False, help='train | test | val', default='val')
    parser.add_argument('--no_crop', required=False, help='True | False', action='store_true')
    parser.add_argument('--no_flip', required=False, help='True | False', action='store_true')
    parser.add_argument('--crop_size', required=False, help='crop size', default=256, type=int)
    parser.add_argument('--scale', required=False, help='scale', default=286, type=int)
    
    parser.print_help()

    return parser

