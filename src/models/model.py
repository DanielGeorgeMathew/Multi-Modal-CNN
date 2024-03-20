import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.resnet import ResNet18, ResNet34, ResNet50
from src.models.rgb_depth_fusion import SqueezeAndExciteFusionAdd
from src.models.context_modules import get_context_module
from src.models.resnet import BasicBlock, NonBottleneck1D
from src.models.model_utils import ConvBNAct, Swish, Hswish
from src.models.convnext import ConvNextEncoder


class SegNet(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=37,
                 encoder_rgb='resnet34',
                 encoder_depth='resnet34',
                 encoder_block='BasicBlock',
                 channels_decoder=None,  # default: [128, 128, 128]
                 pretrained_on_imagenet=False,
                 pretrained_dir='',
                 activation='relu',
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=None,  # default: [1, 1, 1]
                 fuse_depth_in_rgb_encoder='SE-add',
                 upsampling='bilinear'):

        super(SegNet, self).__init__()
        if encoder_depth is None:
            encoder_depth = encoder_rgb
        self.encoder_type = encoder_rgb
        if channels_decoder is None:
            channels_decoder = [512, 256, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        self.fuse_depth_in_rgb_encoder = fuse_depth_in_rgb_encoder

        # set activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()

        """  arch_settings = {
                'atto': {
                    'depths': [2, 2, 6, 2],
                    'channels': [40, 80, 160, 320]
                },
                'femto': {
                    'depths': [2, 2, 6, 2],
                    'channels': [48, 96, 192, 384]
                },
                'pico': {
                    'depths': [2, 2, 6, 2],
                    'channels': [64, 128, 256, 512]
                },
                'nano': {
                    'depths': [2, 2, 8, 2],
                    'channels': [80, 160, 320, 640]
                },
                'tiny': {
                    'depths': [3, 3, 9, 3],
                    'channels': [96, 192, 384, 768]
                },
                'small': {
                    'depths': [3, 3, 27, 3],
                    'channels': [96, 192, 384, 768]
                },
                'base': {
                    'depths': [3, 3, 27, 3],
                    'channels': [128, 256, 512, 1024]
                },
                'large': {
                    'depths': [3, 3, 27, 3],
                    'channels': [192, 384, 768, 1536]
                },
                'xlarge': {
                    'depths': [3, 3, 27, 3],
                    'channels': [256, 512, 1024, 2048]
                },
                'huge': {
                    'depths': [3, 3, 27, 3],
                    'channels': [352, 704, 1408, 2816]
                }
            }"""
        # rgb encoder
        if encoder_rgb == 'resnet18':
            self.encoder_rgb = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet34':
            self.encoder_rgb = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet50':
            self.encoder_rgb = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation)
        elif encoder_rgb == 'convnext_atto':
            self.encoder_rgb = ConvNextEncoder(blocks_per_stage=[2, 2, 6, 2], channels_per_stage=[40, 80, 160, 320])
        elif encoder_rgb == 'convnext_femto':
            self.encoder_rgb = ConvNextEncoder(blocks_per_stage=[2, 2, 6, 2], channels_per_stage=[48, 96, 192, 384])
        elif encoder_rgb == 'convnext_pico':
            self.encoder_rgb = ConvNextEncoder(blocks_per_stage=[2, 2, 6, 2], channels_per_stage=[64, 128, 256, 512])
        elif encoder_rgb == 'convnext_nano':
            self.encoder_rgb = ConvNextEncoder(blocks_per_stage=[2, 2, 8, 2], channels_per_stage=[80, 160, 320, 640])
        elif encoder_rgb == 'convnext_tiny':
            self.encoder_rgb = ConvNextEncoder(blocks_per_stage=[3, 3, 9, 3], channels_per_stage=[96, 192, 384, 768])
        elif encoder_rgb == 'convnext_small':
            self.encoder_rgb = ConvNextEncoder(blocks_per_stage=[3, 3, 27, 3], channels_per_stage=[96, 192, 384, 768])
        elif encoder_rgb == 'convnext_base':
            self.encoder_rgb = ConvNextEncoder(blocks_per_stage=[3, 3, 27, 3], channels_per_stage=[128, 256, 512, 1024])
        elif encoder_rgb == 'convnext_large':
            self.encoder_rgb = ConvNextEncoder(blocks_per_stage=[3, 3, 27, 3], channels_per_stage=[192, 384, 768, 1536])
        elif encoder_rgb == 'convnext_xlarge':
            self.encoder_rgb = ConvNextEncoder(blocks_per_stage=[3, 3, 27, 3],
                                               channels_per_stage=[256, 512, 1024, 2048])
        elif encoder_rgb == 'convnext_huge':
            self.encoder_rgb = ConvNextEncoder(blocks_per_stage=[3, 3, 27, 3],
                                               channels_per_stage=[352, 704, 1408, 2816])
        else:
            raise NotImplementedError(
                'Only ResNets and Convnexts are supported for '
                'encoder_rgb. Got {}'.format(encoder_rgb))

        # depth encoder
        if encoder_depth == 'resnet18':
            self.encoder_depth = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_depth == 'resnet34':
            self.encoder_depth = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_depth == 'resnet50':
            self.encoder_depth = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation,
                input_channels=1)
        elif encoder_depth == 'convnext_atto':
            self.encoder_depth = ConvNextEncoder(input_channels=1, blocks_per_stage=[2, 2, 6, 2],
                                                 channels_per_stage=[40, 80, 160, 320])
        elif encoder_depth == 'convnext_femto':
            self.encoder_depth = ConvNextEncoder(input_channels=1, blocks_per_stage=[2, 2, 6, 2],
                                                 channels_per_stage=[48, 96, 192, 384])
        elif encoder_depth == 'convnext_pico':
            self.encoder_depth = ConvNextEncoder(input_channels=1, blocks_per_stage=[2, 2, 6, 2],
                                                 channels_per_stage=[64, 128, 256, 512])
        elif encoder_depth == 'convnext_nano':
            self.encoder_depth = ConvNextEncoder(input_channels=1, blocks_per_stage=[2, 2, 8, 2],
                                                 channels_per_stage=[80, 160, 320, 640])
        elif encoder_depth == 'convnext_tiny':
            self.encoder_depth = ConvNextEncoder(input_channels=1, blocks_per_stage=[3, 3, 9, 3],
                                                 channels_per_stage=[96, 192, 384, 768])
        elif encoder_depth == 'convnext_small':
            self.encoder_depth = ConvNextEncoder(input_channels=1, blocks_per_stage=[3, 3, 27, 3],
                                                 channels_per_stage=[96, 192, 384, 768])
        elif encoder_depth == 'convnext_base':
            self.encoder_depth = ConvNextEncoder(input_channels=1, blocks_per_stage=[3, 3, 27, 3],
                                                 channels_per_stage=[128, 256, 512, 1024])
        elif encoder_depth == 'convnext_large':
            self.encoder_depth = ConvNextEncoder(input_channels=1, blocks_per_stage=[3, 3, 27, 3],
                                                 channels_per_stage=[192, 384, 768, 1536])
        elif encoder_depth == 'convnext_xlarge':
            self.encoder_depth = ConvNextEncoder(input_channels=1, blocks_per_stage=[3, 3, 27, 3],
                                                 channels_per_stage=[256, 512, 1024, 2048])
        elif encoder_depth == 'convnext_huge':
            self.encoder_depth = ConvNextEncoder(input_channels=1, blocks_per_stage=[3, 3, 27, 3],
                                                 channels_per_stage=[352, 704, 1408, 2816])
        else:
            raise NotImplementedError(
                'Only ResNets and Convnexts are supported for '
                'encoder_depth. Got {}'.format(encoder_depth))
        if 'resnet' in encoder_rgb:
            self.channels_decoder_in = self.encoder_rgb.down_32_channels_out
        else:
            self.channels_decoder_in = self.encoder_depth.channels_per_stage[-1]
        if 'resnet' in encoder_rgb:
            if fuse_depth_in_rgb_encoder == 'SE-add':
                self.se_layer0 = SqueezeAndExciteFusionAdd(
                    64, activation=self.activation)
                self.se_layer1 = SqueezeAndExciteFusionAdd(
                    self.encoder_rgb.down_4_channels_out,
                    activation=self.activation)
                self.se_layer2 = SqueezeAndExciteFusionAdd(
                    self.encoder_rgb.down_8_channels_out,
                    activation=self.activation)
                self.se_layer3 = SqueezeAndExciteFusionAdd(
                    self.encoder_rgb.down_16_channels_out,
                    activation=self.activation)
                self.se_layer4 = SqueezeAndExciteFusionAdd(
                    self.encoder_rgb.down_32_channels_out,
                    activation=self.activation)
        else:
            if fuse_depth_in_rgb_encoder == 'SE-add':
                self.se_layer0 = SqueezeAndExciteFusionAdd(
                    self.encoder_rgb.channels_per_stage[0], activation=self.activation
                )
                self.se_layer1 = SqueezeAndExciteFusionAdd(
                    self.encoder_rgb.channels_per_stage[1], activation=self.activation
                )
                self.se_layer2 = SqueezeAndExciteFusionAdd(
                    self.encoder_rgb.channels_per_stage[2], activation=self.activation
                )
                self.se_layer3 = SqueezeAndExciteFusionAdd(
                    self.encoder_rgb.channels_per_stage[3], activation=self.activation
                )
                self.se_layer4 = SqueezeAndExciteFusionAdd(
                    self.encoder_rgb.channels_per_stage[3], activation=self.activation
                )
        if 'resnet' in encoder_rgb:
            if encoder_decoder_fusion == 'add':
                layers_skip1 = list()
                if self.encoder_rgb.down_4_channels_out != channels_decoder[2]:
                    layers_skip1.append(ConvBNAct(
                        self.encoder_rgb.down_4_channels_out,
                        channels_decoder[2],
                        kernel_size=1,
                        activation=self.activation))
                self.skip_layer1 = nn.Sequential(*layers_skip1)

                layers_skip2 = list()
                if self.encoder_rgb.down_8_channels_out != channels_decoder[1]:
                    layers_skip2.append(ConvBNAct(
                        self.encoder_rgb.down_8_channels_out,
                        channels_decoder[1],
                        kernel_size=1,
                        activation=self.activation))
                self.skip_layer2 = nn.Sequential(*layers_skip2)

                layers_skip3 = list()
                if self.encoder_rgb.down_16_channels_out != channels_decoder[0]:
                    layers_skip3.append(ConvBNAct(
                        self.encoder_rgb.down_16_channels_out,
                        channels_decoder[0],
                        kernel_size=1,
                        activation=self.activation))
                self.skip_layer3 = nn.Sequential(*layers_skip3)
            elif encoder_decoder_fusion == 'None':
                self.skip_layer0 = nn.Identity()
                self.skip_layer1 = nn.Identity()
                self.skip_layer2 = nn.Identity()
                self.skip_layer3 = nn.Identity()
        else:
            if encoder_decoder_fusion == 'add':
                layers_skip1 = list()
                if self.encoder_rgb.channels_per_stage[1] != channels_decoder[2]:
                    layers_skip1.append(ConvBNAct(
                        self.encoder_rgb.channels_per_stage[1],
                        channels_decoder[2],
                        kernel_size=1,
                        activation=self.activation))
                self.skip_layer1 = nn.Sequential(*layers_skip1)

                layers_skip2 = list()
                if self.encoder_rgb.channels_per_stage[2] != channels_decoder[1]:
                    layers_skip2.append(ConvBNAct(
                        self.encoder_rgb.channels_per_stage[2],
                        channels_decoder[1],
                        kernel_size=1,
                        activation=self.activation))
                self.skip_layer2 = nn.Sequential(*layers_skip2)

                layers_skip3 = list()
                if self.encoder_rgb.channels_per_stage[3] != channels_decoder[0]:
                    layers_skip3.append(ConvBNAct(
                        self.encoder_rgb.channels_per_stage[3],
                        channels_decoder[0],
                        kernel_size=1,
                        activation=self.activation))
                self.skip_layer3 = nn.Sequential(*layers_skip3)
            elif encoder_decoder_fusion == 'None':
                self.skip_layer0 = nn.Identity()
                self.skip_layer1 = nn.Identity()
                self.skip_layer2 = nn.Identity()
                self.skip_layer3 = nn.Identity()

        # context modules
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = \
            get_context_module(
                context_module,
                self.channels_decoder_in,
                channels_decoder[0],
                input_size=(height // 32, width // 32),
                activation=self.activation,
                upsampling_mode=upsampling_context_module
            )

        # decoder
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes
        )

    def forward(self, rgb, depth):
        rgb = self.encoder_rgb.forward_first_conv(rgb)
        depth = self.encoder_depth.forward_first_conv(depth)

        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer0(rgb, depth)

        # print(fuse.shape)
        if 'resnet' in self.encoder_type:
            rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
            depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        # block 1
        if 'resnet' in self.encoder_type:
            rgb = self.encoder_rgb.forward_layer1(rgb)
            depth = self.encoder_depth.forward_layer1(depth)
        else:
            rgb, _ = self.encoder_rgb.forward_stage_1(rgb)
            depth, _ = self.encoder_depth.forward_stage_1(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer1(rgb, depth)
        # print(fuse.shape)
        skip1 = self.skip_layer1(fuse)

        # block 2
        if 'resnet' in self.encoder_type:
            rgb = self.encoder_rgb.forward_layer2(rgb)
            depth = self.encoder_depth.forward_layer2(depth)
        else:
            rgb, _ = self.encoder_rgb.forward_stage_2(rgb)
            depth, _ = self.encoder_depth.forward_stage_2(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer2(rgb, depth)
        # print(fuse.shape)

        skip2 = self.skip_layer2(fuse)

        # block 3
        if 'resnet' in self.encoder_type:
            rgb = self.encoder_rgb.forward_layer3(rgb)
            depth = self.encoder_depth.forward_layer3(depth)
        else:
            rgb, _ = self.encoder_rgb.forward_stage_3(rgb)
            depth, _ = self.encoder_depth.forward_stage_3(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer3(rgb, depth)

        # print(fuse.shape)
        skip3 = self.skip_layer3(fuse)
        # rgb = self.encoder_rgb.forward_later3(fuse )
        # block 4
        if 'resnet' in self.encoder_type:
            rgb = self.encoder_rgb.forward_layer4(rgb)
            depth = self.encoder_depth.forward_layer4(depth)
        else:
            rgb = self.encoder_rgb.forward_stage_4(rgb)
            depth = self.encoder_depth.forward_stage_4(depth)

        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer4(rgb, depth)
        # print(fuse.shape)
        # print(fuse.shape)
        out = self.context_module(fuse)
        # print(out.shape)
        # print(skip3.shape)
        # print(skip2.shape)
        # print(skip1.shape)
        out = self.decoder(enc_outs=[out, skip3, skip2, skip1])
        return out


class Decoder(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_decoder,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()

        self.decoder_module_1 = DecoderModule(
            channels_in=channels_in,
            channels_dec=channels_decoder[0],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[0],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_2 = DecoderModule(
            channels_in=channels_decoder[0],
            channels_dec=channels_decoder[1],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[1],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_3 = DecoderModule(
            channels_in=channels_decoder[1],
            channels_dec=channels_decoder[2],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[2],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )
        out_channels = channels_decoder[2]

        self.conv_out = nn.Conv2d(out_channels,
                                  num_classes, kernel_size=3, padding=1)

        # upsample twice with factor 2
        self.upsample1 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)
        self.upsample2 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)

    # torch.Size([1, 512, 20, 15])
    # torch.Size([1, 512, 20, 15])
    # torch.Size([1, 256, 40, 30])
    # torch.Size([1, 128, 80, 60])
    def forward(self, enc_outs):
        enc_out, enc_skip_down_16, enc_skip_down_8, enc_skip_down_4 = enc_outs
        out, out_down_32 = self.decoder_module_1(enc_out, enc_skip_down_16)
        out, out_down_16 = self.decoder_module_2(out, enc_skip_down_8)
        out, out_down_8 = self.decoder_module_3(out, enc_skip_down_4)

        out = self.conv_out(out)
        out = self.upsample1(out)
        out = self.upsample2(out)

        if self.training:
            return out, out_down_8, out_down_16, out_down_32
        return out


class DecoderModule(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_dec,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.encoder_decoder_fusion = encoder_decoder_fusion

        self.conv3x3 = ConvBNAct(channels_in, channels_dec, kernel_size=3,
                                 activation=activation)

        blocks = []
        for _ in range(nr_decoder_blocks):
            blocks.append(NonBottleneck1D(channels_dec,
                                          channels_dec,
                                          activation=activation)
                          )
        self.decoder_blocks = nn.Sequential(*blocks)

        self.upsample = Upsample(mode=upsampling_mode,
                                 channels=channels_dec)
        # for pyramid supervision
        self.side_output = nn.Conv2d(channels_dec,
                                     num_classes,
                                     kernel_size=1)

    def forward(self, decoder_features, encoder_features):
        # print(decoder_features.shape)
        # print(encoder_features.shape)
        out = self.conv3x3(decoder_features)

        out = self.decoder_blocks(out)

        if self.training:
            out_side = self.side_output(out)
        else:
            out_side = None

        out = self.upsample(out)
        if self.encoder_decoder_fusion == 'add':
            out += encoder_features
        return out, out_side


# class Upsample(nn.Module):
#     def __init__(self, mode, channels=None):
#         super(Upsample, self).__init__()
#         self.interp = nn.functional.interpolate
#         if mode == 'bilinear':
#             self.align_corners = False
#         else:
#             self.align_corners = None
#         if 'learned-3x3' in mode:
#             if mode == 'learned-3x3':
#                 self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
#                 self.conv = nn.Conv2d(channels, channels, groups=channels, kernel_size=3, padding=0)
#         elif mode == 'learned-3x3-zeropad':
#             self.pad = nn.Identity()
#             self.conv = nn.Conv2d(channels, channels, groups=channels, kernel_size=3, padding=1)


class Upsample(nn.Module):
    def __init__(self, mode, channels=None):
        super(Upsample, self).__init__()
        self.interp = nn.functional.interpolate

        if mode == 'bilinear':
            self.align_corners = False
        else:
            self.align_corners = None

        if 'learned-3x3' in mode:
            # mimic a bilinear interpolation by nearest neigbor upscaling and
            # a following 3x3 conv. Only works as supposed when the
            # feature maps are upscaled by a factor 2.

            if mode == 'learned-3x3':
                self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=0)
            elif mode == 'learned-3x3-zeropad':
                self.pad = nn.Identity()
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=1)

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
            ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w] * channels))

            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_()

            self.mode = 'nearest'
        else:
            # define pad and conv just to make the forward function simpler
            self.pad = nn.Identity()
            self.conv = nn.Identity()
            self.mode = mode

    def forward(self, x):
        size = (int(x.shape[2] * 2), int(x.shape[3] * 2))
        x = self.interp(x, size, mode=self.mode,
                        align_corners=self.align_corners)
        x = self.pad(x)
        x = self.conv(x)
        return x


# torch.Size([1, 512, 20, 15])
# torch.Size([1, 512, 20, 15])
# torch.Size([1, 256, 40, 30])
# torch.Size([1, 128, 80, 60])

# torch.Size([3, 512, 20, 15])
# torch.Size([3, 512, 40, 30])
# torch.Size([3, 256, 80, 60])
# torch.Size([3, 128, 160, 120])
# torch.Size([3, 512, 40, 30])
# torch.Size([3, 2, 20, 15])
def main():
    height = 640
    width = 480

    model = SegNet(
        height=height,
        width=width,
        encoder_rgb='convnext_tiny',
        encoder_depth='convnext_tiny', num_classes=2)


    print(model)
    model.cuda()
    model.train()
    rgb_image = torch.randn(3, 3, height, width).cuda()
    depth_image = torch.randn(3, 1, height, width).cuda()

    with torch.no_grad():
        output = model(rgb_image, depth_image)
    print(output[0].shape)


if __name__ == '__main__':
    main()