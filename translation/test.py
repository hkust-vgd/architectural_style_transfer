"""
Copyright (C) 2022 HKUST VGD Group
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

##################################################
The script is to infer style transfer between images in two domains.
# example usage:
CUDA_VISIBLE_DEVICES=1 python test.py \
--test_root inputs/images \
--mask_root inputs/masks \
-a day \
-b golden \
--output_path results \
--config_fg checkpoints/config_day2golden_fg.yaml \
--config_bg checkpoints/config_day2golden_bg.yaml \
--checkpoint_fg checkpoints/gen_day2golden_fg.pt \
--checkpoint_bg checkpoints/gen_day2golden_bg.pt \
--opt

# Input data structure:
Option 1: (with preprocessed data)
TEST_ROOT
    - class_a
        - XXXX.jpg
    - class_b
        - XXXX.jpg
    - FG
        - class_a
            - XXXX.jpg
        - class_b
            - XXXX.jpg
    - BG
        - class_a
            - XXXX.jpg
        - class_b  
            - XXXX.jpg
MASK_ROOT
    - class_a
        - XXXX.jpg
    - class_b
        - XXXX.jpg

Option 2: (without preprocessed data)
TEST_ROOT
    - FG
        - class_a
            - XXXX.jpg
        - class_b
            - XXXX.jpg
    - BG
        - class_a
            - XXXX.jpg
        - class_b  
            - XXXX.jpg
MASK_ROOT
    - class_a
        - XXXX.jpg
    - class_b
        - XXXX.jpg

If you run without preprocess, 
this script will take some time to preprocess data 
and generate new testing data first (as shown in Option 1).

"""
import argparse
import sys
import os
import time
import shutil

import torch 
from torch.autograd import Variable
from trainer import DOT_Trainer
from data import make_dataset
import numpy as np
from skimage import img_as_float
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.morphology import dilation, disk, square, binary_dilation
from utils import get_config, get_test_data_loaders, write_image
from mask_images import mask_images
sys.path.append('../optimization') # setting path
from blend_opt import blend_optimize

# marge two images with mask
def blend(img1, img2, alpha):
    '''
        img1, img2: 3-channel image tensors, [1, 3, H, W]
        alpha: mask value
    '''
    # alpha blending 
    if img1.is_cuda:
        img1 = img1.cpu()
    if img2.is_cuda:
        img2 = img2.cpu()
    img1 = img1 * alpha
    img2 = img2 * (1.- alpha)
    dst = img1 + img2

    return dst

def get_mask_alpha(mask_path, size, inverse=False):
    '''
        read and get 3-channel mask
        - mask_path: mask image path
        - size: [H,W]
        - inverse: get inversed mask
    '''
    mask_img = imread(mask_path, as_gray=True) # [H, W], value range[0,255]

    #alpha blending 
    mask_img = resize(mask_img, size) #value range[0,1]
    if inverse:
        mask_img = 1. - mask_img
    alpha = np.array([mask_img] * 3)[np.newaxis, :,:,:] #[1, 3, H, W], tensor default use float32

    return alpha


if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_root', type=str, required=True, help="Path to the root folder of testing images (ex. TEST_ROOT/class_a, class_b)")
    parser.add_argument('--mask_root', type=str, required=True, help="Path to the root folder of testing image masks (ex. MASK_ROOT/class_a, class_b)")
    parser.add_argument('--class_a','-a', type=str, default='day')
    parser.add_argument('--class_b','-b', type=str, default='golden')
    parser.add_argument('--output_path', type=str, default='./results', help="Path to save results")

    parser.add_argument('--config_fg', type=str, help="Path to the config file for foreground model")
    parser.add_argument('--config_bg', type=str, help="Path to the config file for background model")
    parser.add_argument('--checkpoint_fg', type=str, help="Path to load foreground model checkpoint")
    parser.add_argument('--checkpoint_bg', type=str, help="Path to load background model checkpoint")
    parser.add_argument('--a2b', type=int, default=1, help="1 for a2b and others for b2a")
    parser.add_argument('--dilate_kernel', type=int, default=10, help="Kernel size for dilate the mask. Necessary for better blended results.")
    parser.add_argument('--new_size', type=int, default=512, help="new size to resize short side (e.g., 256, 512, 1024)")
    parser.add_argument('--blend_opt', '--opt', action="store_true", default=False, help="Apply blending optimzation in new_size")

    opts = parser.parse_args()

    # Set up CPU/GPU device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # Output directory
    if not os.path.exists(opts.output_path):
        os.makedirs(opts.output_path)

    # Load model setting
    config_fg = get_config(opts.config_fg)
    model_fg = DOT_Trainer(config_fg)
    config_bg = get_config(opts.config_bg)
    model_bg = DOT_Trainer(config_bg)

    # Load models
    state_dict = torch.load(opts.checkpoint_fg)
    model_fg.gen_a.load_state_dict(state_dict['a'])
    model_fg.gen_b.load_state_dict(state_dict['b'])
    model_fg.cuda()
    model_fg.eval()
    encode_fg = model_fg.gen_a.encode if opts.a2b else model_fg.gen_b.encode
    style_encode_fg = model_fg.gen_b.encode if opts.a2b else model_fg.gen_a.encode
    decode_fg = model_fg.gen_b.decode if opts.a2b else model_fg.gen_a.decode

    state_dict = torch.load(opts.checkpoint_bg)
    model_bg.gen_a.load_state_dict(state_dict['a'])
    model_bg.gen_b.load_state_dict(state_dict['b'])
    model_bg.cuda()
    model_bg.eval()
    encode_bg = model_bg.gen_a.encode if opts.a2b else model_bg.gen_b.encode
    style_encode_bg = model_bg.gen_b.encode if opts.a2b else model_bg.gen_a.encode
    decode_bg = model_bg.gen_b.decode if opts.a2b else model_bg.gen_a.decode

    print("--- Loaded models: %.4f seconds ---" % (time.time() - start_time))
    start_time = time.time()

    # Load all images in test folders

    class_src = opts.class_a if opts.a2b else opts.class_b
    class_tgt = opts.class_b if opts.a2b else opts.class_a
    content_paths = sorted(make_dataset(os.path.join(opts.test_root, class_src))) # optional. source paths
    mask_paths = sorted(make_dataset(os.path.join(opts.mask_root, class_src))) # source mask paths
    style_paths = sorted(make_dataset(os.path.join(opts.test_root, class_tgt))) # optional. style paths
    
    # processing with mask
    if (not os.path.exists(os.path.join(opts.test_root,'FG', opts.class_a))) or (not os.path.exists(os.path.join(opts.test_root,'FG', opts.class_b)))\
     or (not os.path.exists(os.path.join(opts.test_root,'BG', opts.class_a))) or (not os.path.exists(os.path.join(opts.test_root,'BG', opts.class_b))):
        style_mask_paths = sorted(make_dataset(os.path.join(opts.mask_root, class_tgt))) # style mask paths
        assert (len(content_paths) == len(mask_paths)) and (len(style_paths) == len(style_mask_paths))

        mask_images(content_paths, mask_paths, opts.test_root, class_src, kernel_size=opts.dilate_kernel)
        mask_images(style_paths, style_mask_paths, opts.test_root, class_tgt, kernel_size=0)

    new_size = opts.new_size
    loader_content_fg, loader_style_fg = get_test_data_loaders(os.path.join(opts.test_root, 'FG'), opts.a2b, opts.class_a, opts.class_b, new_size=new_size)
    loader_content_bg, loader_style_bg = get_test_data_loaders(os.path.join(opts.test_root, 'BG'), opts.a2b, opts.class_a, opts.class_b, new_size=new_size)

    print("--- Loaded and preprocessed data: %.4f seconds ---" % (time.time() - start_time))
    start_time = time.time()


    for idx1, (img1_fg, img1_bg, mask_path) in enumerate(zip(loader_content_fg, loader_content_bg, mask_paths)):
        # Stylization for each source image

        print("%d/%d"%(idx1+1, len(loader_content_fg)))
        
        test_saver_path = os.path.join(opts.output_path, str(idx1))  
        if not os.path.exists(test_saver_path):
            os.mkdir(test_saver_path)
        
        # Optional. save source image
        shutil.copyfile(content_paths[idx1], os.path.join(test_saver_path, 'input.jpg'))

        # Get encoded content feature of source image        
        img1_fg = img1_fg.cuda()
        share_content_fg, _, content_fg, _ = encode_fg(img1_fg)   
        img1_bg = img1_bg.cuda()
        share_content_bg, _, content_bg, _ = encode_bg(img1_bg) 

        for idx2, (img2_fg, img2_bg) in enumerate(zip(loader_style_fg, loader_style_bg)): 
            # Stylize source image with each target style reference
            
            # Optional. save style image
            if idx1 == 0:
                shutil.copyfile(style_paths[idx2], os.path.join(test_saver_path, 'style_{}.jpg'.format(idx2)))

            # Get encoded style feature of target reference
            img2_fg = img2_fg.cuda()
            _, _, _, style_fg = style_encode_fg(img2_fg)
            img2_bg = img2_bg.cuda()
            _, _, _, style_bg = style_encode_bg(img2_bg)

            # Generate stylized image
            with torch.no_grad():
                outputs_fg = decode_fg(share_content_fg, content_fg, style_fg)
                outputs_bg = decode_bg(share_content_bg, content_bg, style_bg)
           
            # Blend images
            b,c,h,w = outputs_fg.shape #[1, 3, H, W]
            alpha = get_mask_alpha(mask_path, size=(h,w))
            outputs = blend(outputs_fg, outputs_bg, alpha)
            
            # Save stylized image
            path = os.path.join(test_saver_path, 'output_{}_blend.jpg'.format(idx2))
            stylized = write_image(outputs, path) # np array

            # Blending optimization
            if opts.blend_opt:
                src_img = img_as_float(imread(content_paths[idx1])) #[H, W, 3]
                mask_img = img_as_float(imread(mask_paths[idx1], as_gray=True)) # [H, W]
                stylized = stylized.cpu().detach().numpy().transpose(1, 2, 0) #[H, W, 3]
                optimized = blend_optimize(src_img, stylized, mask_img, image_size=new_size)
                # optimized = blend_optimize(src_img, stylized, mask_img, image_size=new_size, origin_res=True) #Optional. restore original high resolution
                path = os.path.join(test_saver_path, 'output_{}_opt.jpg'.format(idx2))
                imsave(path, optimized)

    '''Optional:
      If apply optimization as a post-processing with original high-resolution:
      run command:
        python blend_opt.py --blended_dir OUTPUT_DIR --src_dir SOURCE_ORIGIN_RES_DIR --mask_dir MASK_ORIGIN_RES_DIR --origin_res
    '''

    print("--- Time elapsed: %.4f seconds ---" % (time.time() - start_time))


