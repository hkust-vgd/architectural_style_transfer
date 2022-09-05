"""
Copyright (C) 2022 HKUST VGD Group
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# This script is to preprocess data into foreground and background images for data training or testing.
Usage:

python mask_images.py \
--img_dir training_samples/CLASS_NAME \
--mask_dir training_samples/masks/CLASS_NAME \
--class_name CLASS_NAME \
--output_dir training_samples \
--kernel_size 0

python mask_images.py \
--img_dir inputs/images/CLASS_NAME \
--mask_dir inputs/masks/CLASS_NAME \
--class_name CLASS_NAME \
--output_dir inputs/images 
"""

import argparse
import sys
import os
import time
import numpy as np
from data import make_dataset
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.morphology import dilation, disk, square, binary_dilation


def get_mask_alpha(mask_path, size, dilate_kernel=0, inverse=False):
    '''
        read and get 3-channel mask
        - mask_path: mask image path
        - size: [H,W]
        - dilate_kernel: kernel size for dilation
        - inverse: get inversed mask
    '''
    mask_img = imread(mask_path, as_gray=True) # [H, W], value range[0,255]
    mask_img = resize(mask_img, size) #value range[0,1]
    if inverse:
        mask_img = 1. - mask_img
    if dilate_kernel > 0: 
        # mask_img = binary_dilation(mask_img.astype(bool), disk(dilate_kernel)) #Quicker
        mask_img = dilation(mask_img, disk(dilate_kernel))
    alpha = np.transpose([mask_img]*3, (1,2,0)) #[H, W, 3], alpha mask [0,1] 3 channels

    return alpha

def mask_image(img_path, mask_path, kernel_size=3):
    '''
        Get foreground and background images
        Optional: if you want to train with 4 channels, concatinate alpha*255 as 4th channel 
    '''
    img = imread(img_path)
    h,w,c = img.shape
    alpha = get_mask_alpha(mask_path, (h,w), dilate_kernel= kernel_size)
    img_fg = img * alpha
    alpha = get_mask_alpha(mask_path, (h,w), dilate_kernel= kernel_size, inverse=True)
    img_bg = img * alpha

    return img_fg, img_bg

# Testing use
def mask_images(img_paths, mask_paths, output_root, class_name, kernel_size=3):
    output_fg_root = os.path.join(output_root, 'FG', class_name)
    output_bg_root = os.path.join(output_root, 'BG', class_name)
    if not os.path.exists(output_fg_root):
        os.makedirs(output_fg_root)
    if not os.path.exists(output_bg_root):
        os.makedirs(output_bg_root)

    for img_path, mask_path in zip(img_paths, mask_paths):
        img_fg, img_bg = mask_image(img_path, mask_path, kernel_size=kernel_size)
        filename = os.path.basename(img_path)
        imsave(os.path.join(output_fg_root, filename), img_fg)
        imsave(os.path.join(output_bg_root, filename), img_bg)
    print('Preprocessed %d images'%len(img_paths))

# Training use
# Each class folder contains 'train' and 'test' subfolders
def mask_train_images(img_paths, mask_paths, output_root, class_name, kernel_size=3):
    output_fg_root = os.path.join(output_root, 'FG', class_name)
    output_bg_root = os.path.join(output_root, 'BG', class_name)
    if not os.path.exists(os.path.join(output_fg_root, 'train')):
        os.makedirs(os.path.join(output_fg_root, 'train'))
    if not os.path.exists(os.path.join(output_fg_root, 'test')):
        os.makedirs(os.path.join(output_fg_root, 'test'))
    if not os.path.exists(os.path.join(output_bg_root, 'train')):
        os.makedirs(os.path.join(output_bg_root, 'train'))
    if not os.path.exists(os.path.join(output_bg_root, 'test')):
        os.makedirs(os.path.join(output_bg_root, 'test'))

    for img_path, mask_path in zip(img_paths, mask_paths):
        img_fg, img_bg = mask_image(img_path, mask_path, kernel_size=kernel_size)
        set_name = os.path.basename(os.path.dirname(img_path))
        filename = os.path.basename(img_path)
        imsave(os.path.join(output_fg_root, set_name, filename), img_fg)
        imsave(os.path.join(output_bg_root, set_name, filename), img_bg)
    print('Preprocessed %d images'%len(img_paths))


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', '-i', type=str, required=True, help='Path to the images')
    parser.add_argument('--mask_dir', '-m', type=str, required=True,  help="Path to mask folder")
    parser.add_argument('--class_name', '-c', type=str, required=True,  help="Class name {day, golden, blue, night}")
    parser.add_argument('--output_dir', '-out_i', type=str, required=True, help='Path to save mask images')
    parser.add_argument('--kernel_size', '-size', type=int, default=3, help="Optional. Kernel size for dilation. Necessary for better blended results.") 
    opts = parser.parse_args()


    img_paths = sorted(make_dataset(opts.img_dir))
    mask_paths = sorted(make_dataset(opts.mask_dir))
    assert len(img_paths) == len(mask_paths)

    mask_train_images(img_paths, mask_paths, opts.output_dir, opts.class_name, kernel_size=opts.kernel_size)

    print("--- Time elapsed: %.4f sec ---" %((time.time() - start_time)))
