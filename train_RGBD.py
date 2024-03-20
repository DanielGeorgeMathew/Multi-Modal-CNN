import argparse
from datetime import datetime
import json
import pickle
import os
import sys
import time
import random
import warnings
from utils import adjust_learning_rate
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim
from torch.optim.lr_scheduler import OneCycleLR

from src.args import ArgumentParserRGBDSegmentation
from src.models.model import SegNet
from src import utils

from src.utils import load_ckpt
from wound_rgbd import getDataLoaders
import utils
import matplotlib.pyplot as plt

import dropbox
from dropbox.exceptions import AuthError

from zipfile import ZipFile
from glob import glob

# This script can be used to train an RGBD Model on an RGB-Depth Dataset for wound binary segmentation.

def parse_args():
    parser = ArgumentParserRGBDSegmentation(
        description='RGBD Wound Segmentation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    args = parser.parse_args()
    return args


def prepareDatasetPathFiles(args):
    print('Writing train and validation path text files................')

    root_path = 'datasets/Wound-RGBD/Wound-RGBD/finetune/train'
    target_path = 'wound-rgbd-dataset/train.txt'
    file_paths = []
    for modality_path in glob(os.path.join(root_path, 'depth/*.png')):
        rgb_path = modality_path.replace('depth', 'rgb')
        mask_path = modality_path.replace('depth', 'mask')
        if os.path.exists(rgb_path) and os.path.exists(mask_path):
            file_paths.append('{} {} {}\n'.format(rgb_path, modality_path, mask_path))

    with open(target_path, 'w') as fp:
        fp.writelines(file_paths)

    root_path = 'datasets/Wound-RGBD/Wound-RGBD/finetune/val'
    target_path = 'wound-rgbd-dataset/val.txt'
    file_paths = []
    for modality_path in glob(os.path.join(root_path, 'depth/*.png')):
        rgb_path = modality_path.replace('depth', 'rgb')
        mask_path = modality_path.replace('depth', 'mask')
        if os.path.exists(rgb_path) and os.path.exists(mask_path):
            file_paths.append('{} {} {}\n'.format(rgb_path, modality_path, mask_path))

    with open(target_path, 'w') as fp:
        fp.writelines(file_paths)

    args.dataset_dir = os.path.dirname(target_path)


def train_main():
    args = parse_args()
    log_dir = os.path.join(args.results_dir, 'logs')
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=log_dir)
    else:
        log_writer = None

    downloadDropBoxFiles(args)
    if args.dataset_dir == '':
        prepareDatasetPathFiles(args)
    # directory for storing weights and other training related files
    training_starttime = datetime.now().strftime("%d_%m_%Y-%H_%M_%S-%f")
    ckpt_dir = os.path.join(args.results_dir, 'weights',
                            f'checkpoints_{training_starttime}')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, 'confusion_matrices'), exist_ok=True)

    with open(os.path.join(ckpt_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    with open(os.path.join(ckpt_dir, 'argsv.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    # when using multi-scale supervision mask needs to be downsampled.
    label_downsampling_rates = [8, 16, 32]

    # data preparation --------------
    # data_loaders = prepare_data(args, ckpt_dir)
    train_loader, valid_loader = getDataLoaders(args)
    if isinstance(args.nr_decoder_blocks, int):
        nr_decoder_blocks = [args.nr_decoder_blocks] * 3
    elif len(args.nr_decoder_blocks) == 1:
        nr_decoder_blocks = args.nr_decoder_blocks * 3
    else:
        nr_decoder_blocks = args.nr_decoder_blocks
        assert len(nr_decoder_blocks) == 3
    if 'decreasing' in args.decoder_channels_mode:
        if args.decoder_channels_mode == 'decreasing':
            channels_decoder = [512, 256, 128]
    else:
        channels_decoder = [args.channels_decoder] * 3
    # model building -----------------------------------------------------------
    model = SegNet(
        height=args.height,
        width=args.width,
        num_classes=1,
        pretrained_on_imagenet=args.pretrained_on_imagenet,
        pretrained_dir=args.pretrained_dir,
        encoder_rgb=args.encoder,
        encoder_depth=args.encoder_depth,
        encoder_block=args.encoder_block,
        activation=args.activation,
        encoder_decoder_fusion=args.encoder_decoder_fusion,
        context_module=args.context_module,
        nr_decoder_blocks=nr_decoder_blocks,
        channels_decoder=channels_decoder,
        fuse_depth_in_rgb_encoder=args.fuse_depth_in_rgb_encoder,
        upsampling=args.upsampling
    )
    # checking for gpu
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print('Device:', device)
    model.to(device)
    print(model)

    # Checking if finetuning or train from scratch
    if args.finetune is not None:
        checkpoint = torch.load(args.finetune)
        model.load_state_dict(checkpoint['state_dict'])
        print(f'Loaded weights for finetuning: {args.finetune}')

        # print('Freeze the encoder(s).')
        for name, param in model.named_parameters():
            if 'encoder_rgb' in name or 'encoder_depth' in name or 'se_layer' in name:
                param.requires_grad = False

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable Model Parameters: {}".format(n_parameters))
    if args.freeze > 0:
        print('Freeze everything but the output layer(s).')
        for name, param in model.named_parameters():
            if 'out' not in name:
                param.requires_grad = False

    # loss, optimizer, learning rate scheduler

    pos_wght = torch.Tensor([
                                81.83]).cuda()  # Adjust this weight based on ratio of wound pixels to non wound pixels. Otherwise model will not converge and result in Nan/Inf loss
    criterion = torch.nn.BCEWithLogitsLoss(torch.Tensor([81.83]).cuda())

    optimizer = get_optimizer(args, model)

    # in this script lr_scheduler.step() is only called once per epoch
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=[i['lr'] for i in optimizer.param_groups],
        total_steps=args.epochs,
        div_factor=25,
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=1e4
    )

    # load checkpoint if parameter last_ckpt is provided
    if args.last_ckpt:
        ckpt_path = os.path.join(ckpt_dir, args.last_ckpt)
        epoch_last_ckpt, best_miou, best_miou_epoch = \
            load_ckpt(model, optimizer, ckpt_path, device)

        start_epoch = epoch_last_ckpt + 1
    else:
        start_epoch = 0
        best_miou = 0
        best_miou_epoch = 0

    # start training -----------------------------------------------------------
    max_miou = 0.0
    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    num_training_steps_per_epoch = len(train_loader)
    for epoch in range(0, args.epochs):
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        model.train(True)
        header = 'Epoch: [{}]'.format(epoch)
        print(header)
        update_freq = 1
        optimizer.zero_grad()
        avg_train_loss = 0
        avg_train_miou = 0
        for data_iter_step, samples in enumerate(train_loader):
            if data_iter_step % update_freq == 0:
                adjust_learning_rate(optimizer, data_iter_step / len(train_loader) + epoch, args)
            print('Iter {}/{}: '.format(data_iter_step, len(train_loader)))
            images = samples['image'].to(device, non_blocking=True)
            targets = samples['mask'].to(device, non_blocking=True)
            depths = samples['depth'].to(device, non_blocking=True)
            valid_mask = samples['val_mask'].to(device, non_blocking=True)
            # print(images.shape)
            # print(depths.shape)
            # exit()
            output, _, _, _ = model(images, depths)
            if args.nb_classes == 2:
                output = output[:, 0, :, :]
            # fig, ax = plt.subplots(1, 3)
            # ax[0].imshow(torch.nn.Sigmoid()(output[0]).detach().cpu().numpy())
            # ax[1].imshow(targets[0].detach().cpu().numpy())
            # ax[2].imshow(images[0].permute((1, 2, 0)).detach().cpu().numpy())
            # plt.show()
            loss = criterion(output, targets)
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                assert math.isfinite(loss_value)

            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
            torch.cuda.synchronize(device)
            if args.nb_classes == 2:
                miou = utils.evaluate_binary_segmentation(output, targets.long())
            avg_train_loss += loss_value
            avg_train_miou += miou
            min_lr = 10.
            max_lr = 0.

            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])
            print('Train-Loss : {:.3f}, Train-mIoU : {:.2f}'.format(loss_value, miou))

        avg_train_loss /= len(train_loader)
        avg_train_miou /= len(train_loader)
        if log_writer is not None:
            log_writer.update(loss=avg_train_loss, head="train", step=epoch)
            log_writer.update(miou=avg_train_miou, head="train", step=epoch)
        print("Averaged Training Loss: {:.3f}, Averaged Training mIoU: {:.3f}".format(avg_train_loss, avg_train_miou))

        if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            utils.save_model(
                args=args, output_dir=ckpt_dir, model=model, optimizer=optimizer, epoch=epoch)

        if valid_loader is not None:
            avg_val_loss = 0
            avg_val_miou = 0
            with torch.no_grad():
                header = 'Validation:'
                # switch to evaluation mode
                print(header)
                model.eval()
                for data_iter_step, samples in enumerate(valid_loader):
                    print('Iter {}/{}: '.format(data_iter_step, len(valid_loader)))
                    images = samples['image'].to(device, non_blocking=True)
                    targets = samples['mask'].to(device, non_blocking=True)
                    depths = samples['depth'].to(device, non_blocking=True)

                    output = model(images, depths)

                    if args.nb_classes == 2:
                        output = output[:, 0, :, :]

                    loss = criterion(output, targets)
                    loss_value = loss.item()
                    if args.nb_classes == 2:
                        miou = utils.evaluate_binary_segmentation(output, targets.long())
                    print('Val-Loss : {:.3f}, Val-mIoU : {:.2f} '.format(loss_value, miou))
                    avg_val_loss += loss_value
                    avg_val_miou += miou
                    random_index = random.randint(0, images.shape[0] - 1)
                    fig, ax = plt.subplots(1, 4, )
                    ax[0].imshow(images[random_index].permute((1, 2, 0)).detach().cpu().numpy())
                    ax[1].imshow(depths[random_index].squeeze().detach().cpu().numpy())
                    ax[2].imshow((torch.nn.Sigmoid()(output[random_index]) > 0.5).detach().cpu().numpy())
                    ax[3].imshow(targets[random_index].squeeze().detach().cpu().numpy())

                    if log_writer is not None:
                        log_writer.update_figure(head='val', fig=fig,
                                                 step=epoch * num_training_steps_per_epoch + data_iter_step)

                    fig, ax = plt.subplots(1, 4)
                    ax[0].imshow(images[random_index].permute((1, 2, 0)).detach().cpu().numpy())
                    ax[1].imshow(depths[random_index].squeeze().detach().cpu().numpy())
                    ax[2].imshow((torch.nn.Sigmoid()(output[random_index]) > 0.5))
                avg_val_loss /= len(valid_loader)
                avg_val_miou /= len(valid_loader)
                print("Averaged Validation Loss: {:.3f}, Averaged Validation mIoU: {:.3f}".format(avg_val_loss,
                                                                                                  avg_val_miou))

                if max_miou < avg_val_miou:
                    max_miou = avg_val_miou

                    utils.save_model(
                        args=args, output_dir=ckpt_dir, model=model, optimizer=optimizer, epoch="best")
                print('Max Validation mIoU: {:.2f}'.format(max_miou))

                if log_writer is not None:
                    log_writer.update(miou=avg_val_miou, head="val", step=epoch)
                    log_writer.update(loss=avg_val_loss, head="val", step=epoch)

                log_stats = {'train_loss': avg_train_loss,
                             'train_miou': avg_train_miou,
                             'val_loss': avg_val_loss,
                             'val_miou': avg_val_miou,
                             'epoch': epoch,
                             'n_parameters': n_parameters}

        if log_writer is not None:
            log_writer.flush()
        with open(os.path.join(log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def get_optimizer(args, model):
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=True
        )
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )

    print('Using {} as optimizer'.format(args.optimizer))
    return optimizer


if __name__ == '__main__':
    train_main()