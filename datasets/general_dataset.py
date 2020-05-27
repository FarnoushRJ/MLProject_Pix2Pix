from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, Lambda
from PIL import Image
import os
from torch import cat
import numpy as np


class GeneralDataset(Dataset):
    IMG_EXTENSION = '.jpg'

    def __init__(self,
                 dataset_name: str,
                 root: str,
                 sub_folder: str,
                 no_crop: bool,
                 no_flip: bool,
                 crop_size: int,
                 scale: int,
                 ):

        self.__name = dataset_name
        self.__root = root
        self.__sub_folder = sub_folder
        self.__no_crop = no_crop
        self.__no_flip = no_flip
        self.__crop_size = crop_size
        self.__scale = scale
        self.__crop_pos = None
        self.__images = []
        self.__images_path = []

        for i, filename in enumerate(os.listdir(os.path.join(self.__root,
                                                             self.__sub_folder))):
            if filename.endswith(GeneralDataset.IMG_EXTENSION):
                self.__images.append(filename)

    def __len__(self):
        return len(self.__images)

    def __getitem__(self, idx):
        image_name = self.__images[idx]
        image_path = os.path.join(os.path.join(self.__root, self.__sub_folder),
                                  image_name)
        self.__images_path += [image_path]
        image = Image.open(image_path)

        # separate the input and target images
        width, height = image.size
        new_width = int(width / 2)
        image_x = image.crop((0, 0, new_width, height))
        image_y = image.crop((new_width, 0, width, height))

        if not self.__no_crop:
            # specify the crop position
            x = np.random.randint(0, np.maximum(0, self.__scale - self.__crop_size))
            y = np.random.randint(0, np.maximum(0, self.__scale - self.__crop_size))

            self.__crop_pos = (x, y)

        # apply the transformations on both input and target images
        transform = self.get_transforms()

        image_x = transform(image_x)
        image_y = transform(image_y)
        image_xy = cat([image_x, image_y], 2)

        return image_xy

    def get_transforms(self):

        transformations = [Resize(self.__scale, Image.BILINEAR)]

        if not self.__no_crop:
            transformations += [
                Lambda(lambda image: GeneralDataset.__crop(image,
                                                           self.__crop_pos,
                                                           self.__crop_size))
            ]

        if not self.__no_flip and np.random.random() < 0.5:
            transformations.append(Lambda(lambda image: image.transpose(Image.FLIP_LEFT_RIGHT)))

        transformations.extend([
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        return Compose(transformations)

    @staticmethod
    def __crop(image, crop_pos, crop_size):
        old_w, old_h = image.size
        x1, y1 = crop_pos
        new_w = new_h = crop_size
        if old_w > new_w or old_h > new_h:
            return image.crop((x1, y1, x1 + new_w, y1 + new_h))
