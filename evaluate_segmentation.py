import os
import torch
import cv2
from PIL import Image
from src.models.model import SegNet
import numpy as np
import matplotlib.pyplot as plt
import PIL
from wound_rgbd import getDataSet
from tqdm import tqdm
from sklearn.metrics import jaccard_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_binary_segmentation(pred, target):
    out_probs = torch.nn.Sigmoid()(pred)
    pred = (out_probs > 0.9).long()
    smooth = 0.00001
    pred_f = pred.cpu().flatten()
    target_f = target.cpu().flatten()

    TP = torch.sum(pred_f * target_f)
    FP = torch.sum(torch.logical_and(pred_f == 1, target_f == 0))
    FN = torch.sum(torch.logical_and(pred_f == 0, target_f == 1))
    TN = torch.sum(torch.logical_and(pred_f == 0, target_f == 0))

    dice_score = (2. * TP + smooth) / ((torch.sum(target_f) + torch.sum(pred_f)) + smooth)
    miou = jaccard_score(target_f, pred_f)
    precision = TP/(TP + FP + smooth)
    recall = TP/(TP + FN)
    pixel_accuracy = (TP + TN)/(TP + TN + FP + FN)
    return miou, dice_score, precision, recall, pixel_accuracy

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
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable Model Parameters: {}".format(n_parameters))

model.eval()
model.cuda()



val_dataset = getDataSet()
n = len(val_dataset)


avg_miou = 0
avg_dice = 0
avg_precision = 0
avg_recall = 0
avg_accuracy = 0

for i in tqdm(val_dataset):
    image = i['image'].to(device, non_blocking=True).cuda()
    target = i['mask'].to(device, non_blocking=True)
    depth = i['depth'].to(device, non_blocking=True).cuda()
    out = model(image.unsqueeze(0), depth.unsqueeze(0)).squeeze()

    miou, dice, precision, recall, acc = evaluate_binary_segmentation(out, target)

    avg_miou += miou
    avg_dice += dice
    avg_precision += precision
    avg_recall += recall
    avg_accuracy += acc

avg_miou /= n
avg_dice /= n
avg_precision /= n
avg_recall /= n
avg_accuracy /= n
print(avg_miou, avg_dice, avg_precision, avg_recall, avg_accuracy)