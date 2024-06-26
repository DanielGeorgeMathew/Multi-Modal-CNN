from sklearn.utils import resample
import torch
import numpy as np
from PIL import Image, ImageOps
import collections

try:
    import accimage
except ImportError:
    accimage = None
import random
import scipy.ndimage as ndimage
import PIL
import pdb
from kornia.filters.motion import MotionBlur
from torchvision.transforms import RandomRotation, Compose, ColorJitter, Grayscale
import cv2


# random.seed(5)
# This file contains all the pytorch transformations for image augmentation, data preprocessing, etc for RGBD dataset

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class RandomMotionBlur(object):
    def __init__(self, kernel_size=3, angle=35, direction=0.5, border_type='constant'):
        self.motion_blur = MotionBlur(kernel_size, angle, direction, border_type)

    def __call__(self, sample):
        image, mask, depth = sample['image'], sample['mask'], sample['depth']
        if random.random() > 0.5:
            image = self.motion_blur(image.unsqueeze(0)).squeeze(0)
        return {'image': image, 'mask': mask, 'depth': depth}


# class RandomRotate(object):
#     """Random rotation of the image from -angle to angle (in degrees)
#     This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
#     angle: max angle of the rotation
#     interpolation order: Default: 2 (bilinear)
#     reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
#     diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
#     """
#
#     def __init__(self, angle, diff_angle=0, order=2, reshape=False):
#         self.angle = angle
#         self.reshape = reshape
#         self.order = order
#
#     def __call__(self, sample):
#         image, mask = sample['image'], sample['mask']
#
#         applied_angle = random.uniform(-self.angle, self.angle)
#         angle1 = applied_angle
#         angle1_rad = angle1 * np.pi / 180
#
#         image = ndimage.interpolation.rotate(
#             image, angle1, reshape=self.reshape, order=self.order)
#         mask = ndimage.interpolation.rotate(
#             mask, angle1, reshape=self.reshape, order=self.order, mode='nearest')
#
#         image = Image.fromarray(image)
#         mask = Image.fromarray(mask)
#
#         return {'image': image, 'mask': mask}
class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order

    def __call__(self, sample):
        image, mask, depth = sample['image'], sample['mask'], sample['depth']

        applied_angle = random.uniform(-self.angle, self.angle)
        # image = Image.fromarray(image)
        # mask = Image.fromarray(mask)
        image_array = np.asarray(image)
        mask_array = np.asarray(mask)
        depth_array = np.asarray(depth)

        image_center = tuple(np.array(image_array.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, applied_angle, 1.0)

        image_array = cv2.warpAffine(image_array, rot_mat, image_array.shape[1::-1], flags=cv2.INTER_LINEAR)
        mask_array = cv2.warpAffine(mask_array, rot_mat, image_array.shape[1::-1], flags=cv2.INTER_NEAREST)
        depth_array = cv2.warpAffine(depth_array, rot_mat, image_array.shape[1::-1], flags=cv2.INTER_LINEAR)
        # image = image.rotate(angle=applied_angle, resample=Image.BILINEAR)
        # mask = mask.rotate(angle=applied_angle, resample=Image.NEAREST)
        # depth = depth.rotate(angle=applied_angle, resample=Image.BILINEAR)
        image = Image.fromarray(image_array)
        mask = Image.fromarray(mask_array)
        depth = Image.fromarray(depth_array)
        return {'image': image, 'mask': mask, 'depth': depth}


class RandomHorizontalFlip(object):

    def __call__(self, sample):
        image, mask, depth = sample['image'], sample['mask'], sample['depth']
        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(mask):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(mask)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(mask)))

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image, 'mask': mask, 'depth': depth}


class RandomGrayScale(object):

    def __call__(self, sample):
        image, mask, depth = sample['image'], sample['mask'], sample['depth']

        # if not _is_pil_image(image):
        #     raise TypeError(
        #         'img should be PIL Image. Got {}'.format(type(image)))
        # if not _is_pil_image(mask):
        #     raise TypeError(
        #         'img should be PIL Image. Got {}'.format(type(mask)))

        if random.random() < 0.5:
            image = Grayscale(3)(image)

        return {'image': image, 'mask': mask, 'depth': depth}


class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, mask, depth = sample['image'], sample['mask'], sample['depth']
        image = self.changeScale(image, self.size)
        depth = self.changeScale(depth, self.size)
        mask = self.changeScale(mask, self.size, Image.NEAREST)
        return {'image': image, 'mask': mask, 'depth': depth}

    def changeScale(self, img, size, interpolation=Image.BILINEAR):
        if not _is_pil_image(img):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(img)))
        if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))
        if img.mode == 'I;16':
            img_array = np.asarray(img)
            img_array = cv2.resize(img_array, size[::-1], interpolation=cv2.INTER_LINEAR)
            img = Image.fromarray(img_array)
            return img
        if isinstance(size, int):
            w, h = img.size
            # h, w = img.shape
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)


class CenterCrop(object):
    def __init__(self, size_image):
        self.size_image = size_image

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # crop image and depth to (304, 228)
        # print(np.unique(depth))
        image = self.centerCrop(image, self.size_image)
        # depth = self.centerCrop(depth, self.size_image)
        # resize depth to (152, 114) downsample 2
        return {'image': image, 'mask': mask}

    def centerCrop(self, image, size):
        w1, h1 = image.size
        tw, th = size
        if w1 == tw and h1 == th:
            return image
        # (320-304) / 2. = 8
        # (240-228) / 2. = 8
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))
        image = image.crop((x1, y1, tw + x1, th + y1))
        return image


class RandomCrop(object):
    def __init__(self, size_image):
        self.size_image = size_image

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        w_crop, h_crop = self.size_image
        w, h = image.size
        left = random.randint(0, w - w_crop)
        top = random.randint(0, h - h_crop)
        image = image.crop((left, top, left + w_crop, top + h_crop))
        mask = mask.crop((left, top, left + w_crop, top + h_crop))
        return {'image': image, 'mask': mask}


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        image, mask, depth = sample['image'], sample['mask'], sample['depth']

        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        # ground truth depth of training samples is stored in 8-bit while test samples are saved in 16 bit
        image = self.to_tensor(image)
        mask = self.to_tensor(mask)
        depth = self.to_tensor(depth)
        # print(image.shape)
        # print(mask.shape)
        # print(depth.shape)
        return {'image': image, 'mask': mask.squeeze(), 'depth': depth}

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # convert image to (0,1)
            return img.float().div(255)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)
        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False) / 65535.0)
        else:
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class Lighting(object):

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if self.alphastd == 0:
            return image

        alpha = image.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(image).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        image = image.add(rgb.view(3, 1, 1).expand_as(image))

        return {'image': image, 'depth': depth}


# class Grayscale(object):
#
#     def __call__(self, img):
#         gs = img.clone()
#         gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
#         gs[1].copy_(gs[0])
#         gs[2].copy_(gs[0])
#         return gs


# class Saturation(object):
#
#     def __init__(self, var):
#         self.var = var
#
#     def __call__(self, img):
#         gs = Grayscale()(img)
#         alpha = random.uniform(-self.var, self.var)
#         return img.lerp(gs, alpha)


# class Brightness(object):
#
#     def __init__(self, var):
#         self.var = var
#
#     def __call__(self, img):
#         gs = img.new().resize_as_(img).zero_()
#         alpha = random.uniform(-self.var, self.var)
#
#         return img.lerp(gs, alpha)


# class Contrast(object):
#
#     def __init__(self, var):
#         self.var = var
#
#     def __call__(self, img):
#         gs = Grayscale()(img)
#         gs.fill_(gs.mean())
#         alpha = random.uniform(-self.var, self.var)
#         return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if self.transforms is None:
            return {'image': image, 'depth': depth}
        order = torch.randperm(len(self.transforms))
        for i in order:
            image = self.transforms[i](image)
        return {'image': image, 'depth': depth}


class RandomColorJitter(object):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transform = ColorJitter(brightness, contrast, saturation)

    def __call__(self, sample):
        image, mask, depth = sample['image'], sample['mask'], sample['depth']
        image = self.transform(image.unsqueeze(0)).squeeze(0)
        return {'image': image, 'mask': mask, 'depth': depth}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        image, mask = sample['image'], sample['mask']
        image = self.normalize(image, self.mean, self.std)

        return {'image': image, 'mask': mask}

    def normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for R, G, B channels respecitvely.
            std (sequence): Sequence of standard deviations for R, G, B channels
                respecitvely.
        Returns:
            Tensor: Normalized image.
        """

        # TODO: make efficient
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor