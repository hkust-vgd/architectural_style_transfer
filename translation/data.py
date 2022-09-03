"""
Copyright (C) 2022 HKUST VGD Group 
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import os.path
import torch.utils.data as data
from PIL import Image
import random
import numpy as np  
import torchvision.transforms as transforms


###########################################################################################
# Modified Code from
##  https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
### So that it supports:
####  - so that this class can load images from both current directory and its subdirectories.
####  - image reading in YCbCr, grayscale
####  - resize with designated shorter side and origianl aspect ratio
###########################################################################################

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.tif',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

# conform to JPEG conversion standard, Y is same as ITU-R BT.601 standard: 
# ref: https://github.com/python-pillow/Pillow/blob/61a35f94cf8a217db3e67d32db943b05e593e781/src/libImaging/ConvertYCbCr.c#L17-L21
#      https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
#      https://en.wikipedia.org/wiki/YCbCr
def default_loader(path):
    '''
        Load image using YCbCr color space
    '''
    return Image.open(path).convert('YCbCr') 

def rgb_loader(path):
    '''
        Load image using RGB color space
    '''
    return Image.open(path).convert('RGB')   

# https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
# Conform to ITU-R BT.601 standard
def gray_loader(path):
    '''
        Load image and convert to gray image
    '''
    return Image.open(path).convert('L') 


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, new_size=None, transform=None, loader=default_loader, base=32):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = sorted(imgs)
        self.transform = transform
        self.loader = loader
        self.new_size = new_size
        self.base = base

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.new_size:
            width, height = img.size
            if width > height:
                new_width = int(width * (self.new_size/height) / self.base + 0.5) * self.base
                img = img.resize((new_width, self.new_size))
            else:
                new_height = int(height * (self.new_size/width) / self.base + 0.5) * self.base

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)

class ImageList(data.Dataset):

    def __init__(self, paths, new_size=None, transform=None, loader=default_loader, base=32):
        imgs = sorted(paths)
        self.imgs = []
        for img in imgs:
            if is_image_file(img): 
                self.imgs.append(img)

        if len(self.imgs) == 0:
            raise(RuntimeError("Found 0 images" + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.transform = transform
        self.loader = loader
        self.new_size = new_size
        self.base = base

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.new_size:
            width, height = img.size
            if width > height: 
                new_width = int(round(width * (self.new_size/height) / self.base)) * self.base
                # new_width = self.new_size
                img = img.resize((new_width, self.new_size))
            else:
                new_height = int(round(height * (self.new_size/width) / self.base)) * self.base
                # new_height= self.new_size
                img = img.resize((self.new_size, new_height))

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)

###########################################################################################
# Modified Code from
##  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/image_folder.py
##  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/data/base_dataset.py 
# So that it supports: 
#### - load image with aligned mask with consistent transformation
#### - resize image with designated shorter size and original aspect ratio
###########################################################################################

class TestAlignedImageFolder(data.Dataset):

    def __init__(self, roots1, roots2, opt=None, new_size=None, transform=None, loader1=rgb_loader, loader2=gray_loader, base=32):
        imgs = []
        for root1 in roots1:
            imgs = imgs + sorted(make_dataset(root1))
        masks = []
        for root2 in roots2:
            masks = masks + sorted(make_dataset(root2)) 
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + roots1 + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        if len(masks) == 0:
            raise(RuntimeError("Found 0 images in: " + roots2 + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS))) 
        # assert (len(imgs) == len(masks)) 

        self.roots1 = roots1
        self.roots2 = roots2 
        self.imgs = imgs
        self.masks = masks 
        self.transform = transform
        self.loader1 = loader1
        self.loader2 = loader2
        self.new_size = new_size
        self.opt = opt 
        self.base = base

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        image_path = self.imgs[index]
        mask_path = self.masks[index]
        img = self.loader1(image_path)
        mask = self.loader2(mask_path)
        if self.new_size:
            width, height = img.size
            if width > height:
                new_width = int(width * (self.new_size/height) / self.base + 0.5) * self.base
                img = img.resize((new_width, self.new_size))
                mask = mask.resize((new_width, self.new_size))
            else:
                new_height = int(height * (self.new_size/width) / self.base + 0.5) * self.base
                img = img.resize((self.new_size, new_height))
                mask = mask.resize((self.new_size, new_height))

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
        else:            
            # apply the same transform to both image and mask
            transform_params = get_params(self.opt, img.size)
            img_transform = get_transform(self.opt, transform_params, grayscale=False)
            mask_transform = get_transform(self.opt, transform_params, grayscale=True)
            img = img_transform(img)
            mask = mask_transform(mask)

        return {'img': img, 'mask': mask}
    

    def __len__(self):
        # number of N images (in a batch)
        return len(self.imgs)

class AlignedImageFolder(data.Dataset):

    def __init__(self, root1, root2, opt, new_size=None, transform=None, loader1=default_loader, loader2=gray_loader, base=32):
        imgs = sorted(make_dataset(root1))
        masks = sorted(make_dataset(root2))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        if len(masks) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        
        self.root1 = root1
        self.root2 = root2
        self.imgs = imgs
        self.masks = masks
        self.transform = transform
        self.loader1 = loader1
        self.loader2 = loader2
        self.new_size = new_size
        self.opt = opt
        self.base = base


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        image_path = self.imgs[index]
        mask_path = self.masks[index]
        img = self.loader1(image_path)
        mask = self.loader2(mask_path)
        if self.new_size:
            width, height = img.size
            if width > height:
                new_width = int(width * (self.new_size/height) / self.base + 0.5) * self.base
                img = img.resize((new_width, self.new_size))
                mask = mask.resize((new_width, self.new_size))
            else:
                new_height = int(height * (self.new_size/width) / self.base + 0.5) * self.base
                img = img.resize((self.new_size, new_height))
                mask = mask.resize((self.new_size, new_height))

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
        else:            
            # apply the same transform to both image and mask
            transform_params = get_params(self.opt, img.size)
            img_transform = get_transform(self.opt, transform_params, grayscale=False)
            mask_transform = get_transform(self.opt, transform_params, grayscale=True)
            img = img_transform(img)
            mask = mask_transform(mask)

        return {'img': img, 'mask': mask}
    

    def __len__(self):
        return len(self.imgs)



def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt['preprocess'] == 'resize_and_crop':
        new_h = new_w = opt['load_size']
    elif opt['preprocess'] == 'scale_width_and_crop':
        new_w = opt['load_size']
        new_h = opt['load_size'] * h // w

    x = random.randint(0, np.maximum(0, new_w - opt['crop_size']))
    y = random.randint(0, np.maximum(0, new_h - opt['crop_size']))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}


def get_transform(opt, params=None, grayscale=False, method=Image.NEAREST, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt['preprocess']:
        osize = [opt['load_size'], opt['load_size']]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt['preprocess']:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt['load_size'], opt['crop_size'], method)))

    if 'crop' in opt['preprocess']:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt['crop_size']))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt['crop_size'])))

    if opt['preprocess'] == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=32, method=method))) 

    if not opt['no_flip']:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

