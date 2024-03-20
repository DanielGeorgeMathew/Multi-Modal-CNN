import os
import torch
import cv2
from PIL import Image
from src.models.model import SegNet
import numpy as np
import matplotlib.pyplot as plt
import PIL

# This file uses the model trained weights to perform RGBD Wound Segmentation on an RGB-Depth Pair of inputs


model = SegNet(
            height=640,
            width=480,
            num_classes=1,
            encoder_rgb='resnet34',
            encoder_depth='resnet34',
            encoder_block='NonBottleneck1D',
            activation='relu',
            encoder_decoder_fusion='add',
            context_module='ppm',
            nr_decoder_blocks=[3]*3,
            channels_decoder=[512, 256, 128],
            fuse_depth_in_rgb_encoder='SE-add',
            upsampling='bilinear'
        )


model.load_state_dict(torch.load(
    '',
    map_location=torch.device('cpu'))['model'])
model.eval()
model.cpu()


img = Image.open('~/Daniel/datasets/RGBD-Segmentation/finetune/val/rgb/B5E19850-CE05-4FF7-936E-F57C915638FE_20.png')
depth = cv2.imread('~/Daniel/datasets/RGBD-Segmentation/finetune/val/depth/B5E19850-CE05-4FF7-936E-F57C915638FE_20.png', cv2.IMREAD_ANYDEPTH)/65535.0
# img = PIL.ImageEnhance.Brightness(img).enhance(2)

# increasing the contrast 20%

# img = PIL.ImageEnhance.Contrast(img).enhance(2)

# img_resized = img.resize((240, 320))

img_tensor = (torch.from_numpy(np.asarray(img)) / 255).permute((2, 0, 1)).unsqueeze(0)
depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).float()
# depth_tensor = torch.zeros_like(depth_tensor)
out = model(img_tensor, depth_tensor)
out_prob = torch.nn.Sigmoid()(out).squeeze()
img_array = np.asarray(img)
pred_mask = cv2.resize((out_prob > 0.9).detach().numpy().astype(img_array.dtype), (480, 640),
                       interpolation=cv2.INTER_NEAREST)
redImg = np.zeros(img_array.shape, img_array.dtype)
redImg[:, :] = (0, 255, 0)
redMask = cv2.bitwise_and(redImg, redImg, mask=pred_mask)
cv2.addWeighted(redMask, 0.3, img_array, 1, 0, img_array)


fig, ax = plt.subplots(1, 2)
ax[0].imshow(img_array)
ax[1].imshow(out_prob.detach().numpy(), cmap='gray')
plt.show()