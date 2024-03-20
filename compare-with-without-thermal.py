import os

import PIL.Image
import torch
from tqdm import tqdm
from glob import glob
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import json
import open3d as o3d
from collections import Counter
import math
import pymeshfix
from src.models.model import SegNet
from src.models.model_one_modality import SegNetOneModality
from PIL import ImageOps

# This script compares 2 segmentation masks from RGB model and RGB-Thermal Model.


model_rgbt = SegNet(height=480,
                    width=640,
                    num_classes=1,
                    pretrained_on_imagenet=False,
                    pretrained_dir='',
                    encoder_rgb='resnet34',
                    encoder_depth='resnet34',
                    encoder_block='NonBottleneck1D',
                    activation='relu',
                    encoder_decoder_fusion='add',
                    context_module='ppm',
                    nr_decoder_blocks=[3]*3,
                    channels_decoder=[512, 256, 128],
                    fuse_depth_in_rgb_encoder='SE-add',
                    upsampling='learned-3x3-zeropad')

model_rgb = SegNetOneModality(height=480,
                              width=640,
                              pretrained_on_imagenet=False,
                              encoder='resnet34',
                              encoder_block='NonBottleneck1D',
                              activation='relu',
                              input_channels=3,
                              encoder_decoder_fusion='add',
                              context_module='ppm',
                              num_classes=1,
                              pretrained_dir='',
                              nr_decoder_blocks=[3]*3,
                              channels_decoder=[512, 256, 128],
                              weighting_in_encoder='SE-add',
                              upsampling='learned-3x3-zeropad')

model_rgbt.load_state_dict(torch.load(
    'path to RGB-Thermal Weights',
    map_location=torch.device('cpu'))['model'])

model_rgb.load_state_dict(torch.load(
    'path to RGB-Depth weights',
    map_location=torch.device('cpu'))['model'])

model_rgbt.eval()
model_rgb.eval()

root_path = '~/results/case_13/day_1/results/scene_1'
img = Image.open(os.path.join(root_path, 'photo.png'))
img = img.resize((640, 480), PIL.Image.BICUBIC)
mask = Image.open(os.path.join(root_path, 'mask.png'))
mask = ImageOps.grayscale(mask)
mask = mask.resize((640,480), PIL.Image.NEAREST)
thermal = cv2.imread(os.path.join(root_path, 'thermal.png'), 0)
thermal = cv2.resize(thermal, (640, 480), interpolation=cv2.INTER_LINEAR)/255.0
img_array = np.asarray(img)

img_tensor = (torch.from_numpy(np.asarray(img)) / 255).permute((2, 0, 1)).unsqueeze(0)
thermal_tensor = torch.from_numpy(thermal).unsqueeze(0).unsqueeze(0).float()

out_rgbt = model_rgbt(img_tensor, thermal_tensor)
out_prob_rgbt = torch.nn.Sigmoid()(out_rgbt).squeeze()
pred_mask_rgbt = cv2.resize((out_prob_rgbt > 0.5).detach().numpy().astype(img_array.dtype), (640, 480),
                       interpolation=cv2.INTER_NEAREST)


out_rgb = model_rgb(img_tensor)
out_prob_rgb = torch.nn.Sigmoid()(out_rgb).squeeze()
pred_mask_rgb = cv2.resize((out_prob_rgb > 0.5).detach().numpy().astype(img_array.dtype), (640, 480),
                       interpolation=cv2.INTER_NEAREST)

fig, ax = plt.subplots(1, 5)
ax[0].imshow(img_array)
ax[0].set_title('RGB Image')
ax[1].imshow(thermal)
ax[1].set_title('Thermal Image')
ax[2].imshow(pred_mask_rgbt)
ax[2].set_title('Mask output from RGB-thermal CNN')
ax[3].imshow(pred_mask_rgb)
ax[3].set_title('Mask output from RGB CNN')
ax[4].imshow(mask)
ax[4].set_title('Ground Truth Mask')
plt.show()