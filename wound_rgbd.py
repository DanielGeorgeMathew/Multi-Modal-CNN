import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torchvision.transforms import Compose
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
import json
import os
import cv2
from PIL import ImageOps
from glob import glob
from transforms import Scale, RandomHorizontalFlip, ToTensor, Normalize, RandomRotate, RandomCrop, RandomMotionBlur, \
    RandomColorJitter, RandomGrayScale
import torch

# This file contains dataloaders for RGBD dataset

# random.seed(5)


class woundFinetuneDataset(Dataset):
    def __init__(self, dataset_paths, transforms):
        self.transforms = transforms
        self.image_paths = dataset_paths['images']
        self.mask_paths = dataset_paths['masks']
        self.depth_paths = dataset_paths['depths']

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        depth_path = self.depth_paths[index]

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        mask = ImageOps.grayscale(mask)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth = Image.fromarray(depth)

        sample = {'image': img, 'mask': mask, 'depth': depth}
        ret = self.transforms(sample)
        ret['val_mask'] = ret['depth'] != 0
        return ret

    def __len__(self):
        return len(self.image_paths)


def getDataLoaders(args):
    with open(os.path.join(args.dataset_dir, 'train.txt'), 'r') as fp:
        train_paths = fp.readlines()

    train_img_paths = [i.rstrip().split(' ')[0] for i in train_paths]
    train_depth_paths = [i.rstrip().split(' ')[1] for i in train_paths]
    train_mask_paths = [i.rstrip().split(' ')[2] for i in train_paths]


    train_dataset = woundFinetuneDataset({'images': train_img_paths,
                                          'masks': train_mask_paths,
                                          'depths': train_depth_paths},
                                         Compose([
                                             RandomHorizontalFlip(),
                                             RandomRotate(angle=30),
                                             Scale((args.height, args.width)),
                                             ToTensor(),
                                             RandomColorJitter(0.4, 0.2, 0.1),
                                             RandomMotionBlur(kernel_size=5),
                                             # RandomGrayScale()
                                         ]))


    with open(os.path.join(args.dataset_dir, 'val.txt'), 'r') as fp:
        val_paths = fp.readlines()

    val_img_paths = [i.rstrip().split(' ')[0] for i in val_paths]
    val_depth_paths = [i.rstrip().split(' ')[1] for i in val_paths]
    val_mask_paths = [i.rstrip().split(' ')[2] for i in val_paths]

    validation_dataset = woundFinetuneDataset({'images': val_img_paths,
                                               'masks': val_mask_paths,
                                               'depths': val_depth_paths},
                                              Compose([Scale((args.height, args.width)),
                                                       ToTensor()]))

    print("Train Set Size: {}, Validation Set Size: {}".format(len(train_dataset), len(validation_dataset)))

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, shuffle=False,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        validation_dataset, shuffle=False,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=False
    )

    return data_loader_train, data_loader_val