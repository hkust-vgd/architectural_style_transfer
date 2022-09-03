"""
Copyright (C) 2022 HKUST VGD Group 
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
import math
import time 

import torch
import torchvision.utils as vutils
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
import numpy as np
import yaml

from data import * #ImageFolder
from skimage import color # color space conversion

'''
Functions: 
- Read configuration
- Testing data loading
- Training data loading
- Image writing
- Model loading and initialization
- Timer
'''

def get_config(config):
    '''
        Read configs
    '''
    with open(config, 'r') as stream:
        # yaml.safe_load(stream) or !pip install pyyaml==5.4.1
        # ref: https://stackoverflow.com/questions/69564817/typeerror-load-missing-1-required-positional-argument-loader-in-google-col
        return yaml.load(stream, Loader=yaml.FullLoader)

################################################################
################# Testing Use Data Loader ######################
################################################################

def get_test_single_data_loader(input_folder, new_size=None, return_dataset=False):
    '''
        input_folder: test folder
        new_size: new size for shorter side to be resized with aspect ratio
        return_dataset: if return whole dataset
    '''
    return get_test_data_loader_folder(input_folder, batch_size=1, new_size=new_size, num_workers=1, return_dataset=return_dataset)

def get_test_data_loaders(path, a2b, class_a, class_b, new_size=None):
    test_loader_a = get_test_data_loader_folder(os.path.join(path, class_a), 1,
                                                new_size=new_size, num_workers=1)
    test_loader_b = get_test_data_loader_folder(os.path.join(path, class_b), 1,
                                                new_size=new_size, num_workers=1)

    if a2b:
        return test_loader_a, test_loader_b
    else:
        return test_loader_b, test_loader_a

def get_random_test_data_loaders(path, new_size=None):
    test_loader = get_test_data_loader_folder(os.path.join(path), 1, new_size=new_size, num_workers=1)
    return test_loader

def get_test_w_mask_data_loaders(path, mask_path, a2b, class_a, class_b, new_size=None):
    test_loader_a = get_test_data_loader_folder(os.path.join(path, class_a), 1,
                                                new_size=new_size, num_workers=1)
    test_loader_b = get_test_data_loader_folder(os.path.join(path, class_b), 1,
                                                new_size=new_size, num_workers=1)
    mask_loader_a = get_test_data_nonorm_loader_folder(os.path.join(mask_path, class_a), 1,
                                                new_size=new_size, num_workers=1, loader=gray_loader) 
    mask_loader_b = get_test_data_nonorm_loader_folder(os.path.join(mask_path, class_b), 1,
                                                new_size=new_size, num_workers=1, loader=gray_loader) 
    if a2b:
        return test_loader_a, test_loader_b, mask_loader_a, mask_loader_b
    else:
        return test_loader_b, test_loader_a, mask_loader_b, mask_loader_b

# Do not apply normalization
def get_test_data_nonorm_loader_folder(input_folder, batch_size, new_size=None, num_workers=1, loader=None):
    transform_list = [transforms.ToTensor()] 
    transform = transforms.Compose(transform_list)
    if loader is not None:
        dataset = ImageFolder(input_folder, new_size=new_size,  transform=transform, loader=loader)
    else:
        dataset = ImageFolder(input_folder, new_size=new_size, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
    return loader

def get_test_data_loader_folder(input_folder, batch_size, new_size=None, num_workers=1, return_dataset=False):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, new_size=new_size, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=num_workers)
    if return_dataset:
        return dataset, loader
    else:
        return loader

def get_test_data_loader_list(paths, batch_size=1, train=False, new_size=None, num_workers=1):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform = transforms.Compose(transform_list) 
    dataset = ImageList(paths, new_size=new_size, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return dataset, loader 

# Particularly for image and mask loader with same image transformation
def get_aligned_test_data_loader_folder(input_folder1, input_folder2, batch_size, train=False, new_size=None,
                                        crop=False, num_workers=1):
    '''
        input_folder1: image folder
        input_folder2: mask folder
    '''
    opt = {
        'preprocess': 'none', # 'resize_and_crop', 'scale_width_and_crop', 'none'
        'no_flip': True
    }
    if train:
        opt['no_flip'] = False
        opt['preprocess'] = 'crop'
    if new_size is not None:
        opt['load_size'] = new_size
        if train:
            opt['preprocess'] = 'resize_and_crop'
        else: 
            opt['preprocess'] = 'scale_width_and_crop'
    if crop or train:
        opt['crop_size'] = width

    dataset = TestAlignedImageFolder(input_folder1, input_folder2, opt, new_size=new_size, transform=None, loader1=default_loader, loader2=gray_loader)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader


################################################################
################# Training Use Data Loader #####################
################################################################

def get_all_data_loaders(conf, class_a, class_b):
    batch_size = conf['batch_size']
    num_workers = conf['num_workers']
    new_size = conf['new_size']
    height = conf['crop_image_height']
    width = conf['crop_image_width']
        
    train_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], class_a, 'train'),
                                            batch_size, True, new_size, height, width, True, num_workers=num_workers)
    test_loader_a = get_data_loader_folder(os.path.join(conf['data_root'], class_a, 'test'),
                                            batch_size, False, new_size, height, width, True, num_workers=num_workers)
    train_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], class_b, 'train'),
                                            batch_size, True, new_size, height, width, True, num_workers=num_workers)
    test_loader_b = get_data_loader_folder(os.path.join(conf['data_root'], class_b, 'test'),
                                            batch_size, False, new_size, height, width, True, num_workers=num_workers)
    return train_loader_a, train_loader_b, test_loader_a, test_loader_b


def get_data_loader_folder(input_folder, batch_size, train, new_size=None,
                           height=256, width=256, crop=True, num_workers=4):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if train or crop else transform_list
    transform_list = [transforms.Resize((new_size, new_size))] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)

    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

def get_aligned_data_loader_folder(input_folder1, input_folder2, batch_size, train=True, new_size=None,
                                    crop=True, num_workers=4):
    opt = {
        'preprocess': 'none', # 'resize_and_crop', 'scale_width_and_crop', 'none'
        'no_flip': True
    }
    if train:
        opt['no_flip'] = False
        opt['preprocess'] = 'crop'
    if new_size is not None:
        opt['load_size'] = new_size
        if train:
            opt['preprocess'] = 'resize_and_crop'
        else: 
            opt['preprocess'] = 'scale_width_and_crop'
    if crop or train:
        opt['crop_size'] = width

    dataset = AlignedImageFolder(input_folder1, input_folder2, opt, new_size=new_size, transform=None, loader1=default_loader, loader2=gray_loader)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

#############################################################
################## Image Writers (YCbcr) ####################
#############################################################
# Eq.(2)
# skimage.color uses ITU-R BT.601 conversion standard: 
# ref: https://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.rgb2ycbcr
#      https://en.wikipedia.org/wiki/YCbCr
def ycbcr2rgb_transpose_mc(img_ycbcr_mc):
    '''
        INPUTS: batch of prediction
            img_ycbcr_mc    (3,H,W) [-1, 1]
        OUTPUTS: 
            grid_rgb    RGB outputs (3,H,W) [0,255]
    '''
    if isinstance(img_ycbcr_mc, Variable):
        img_ycbcr_mc = img_ycbcr_mc.data.cpu()
    if img_ycbcr_mc.is_cuda:
        img_ycbcr_mc = img_ycbcr_mc.cpu()

    assert img_ycbcr_mc.dim()==3, 'only for single input'

    pred_ycbcr = (img_ycbcr_mc+1)/2 # normalize to [0,1]
    grid_ycbcr =  pred_ycbcr.numpy().astype('float64') #numpy array
    grid_rgb = np.clip(color.ycbcr2rgb(grid_ycbcr.transpose(1, 2, 0)*255), 0, 1) #[0,255]=>RGB=>clip to [0,1]   
    return torch.from_numpy(grid_rgb.transpose((2, 0, 1)))

# Eq.(2)
# skimage.color uses ITU-R BT.601 conversion standard: 
# ref: https://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.ycbcr2rgb
#      https://en.wikipedia.org/wiki/YCbCr
def batch_ycbcr2rgb_transpose_mc(img_ycbcr_mc, display_image_num):
    '''
        INPUTS: batch of prediction
            img_ycbcr_mc    (B,3,H,W) [-1, 1]
        OUTPUTS: 
            grid_rgb    RGB outputs (3,H,W) [0,255]
    '''
    if isinstance(img_ycbcr_mc, Variable):
        img_ycbcr_mc = img_ycbcr_mc.data.cpu()
    if img_ycbcr_mc.is_cuda:
        img_ycbcr_mc = img_ycbcr_mc.cpu()

    assert img_ycbcr_mc.dim()==4, 'only for batch input'
    grid_ycbcr = vutils.make_grid(img_ycbcr_mc, nrow=display_image_num, padding=0, normalize=True).numpy().astype('float64') #[0,1]
    grid_rgb = np.clip(color.ycbcr2rgb(grid_ycbcr.transpose(1, 2, 0)*255), 0, 1) #[0,255]=>RGB=>clip to [0,1] 

   
    return torch.from_numpy(grid_rgb.transpose((2, 0, 1)))

def write_image(image_output, file_name):
    '''
        transfer a Ycbcr tensor to an RGB image
        save image in file_name (path+name)
    '''
    image = ycbcr2rgb_transpose_mc(image_output.squeeze(0))
    vutils.save_image(image.data, file_name, padding=0, normalize=True)

    return image #RGB

def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs] # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = batch_ycbcr2rgb_transpose_mc(image_tensor, display_image_num)  
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    __write_images(image_outputs[0:n//2], display_image_num, '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[n//2:n], display_image_num, '%s/gen_b2a_%s.jpg' % (image_directory, postfix))


def write_loss(iterations, trainer, train_writer):
    members = [attr for attr in dir(trainer) \
               if not callable(getattr(trainer, attr)) and not attr.startswith("__") and ('loss' in attr or 'grad' in attr or 'nwd' in attr)]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)

######################################################
################## Model Loader ######################
######################################################
# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


######################
### Training Timer ###
######################
class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))
