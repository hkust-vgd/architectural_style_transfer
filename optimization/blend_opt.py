"""
Copyright (C) 2022 HKUST VGD Group
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

Code is modified from https://github.com/wuhuikai/GP-GAN
- We use our blended translated results as style constraint with out any smoothing.
- It support iteractive optimization with different color weights.

# example usage:
python blend_opt.py \
--blended_dir ../translation/results/ \
--src_dir ../translation/inputs/images/day/ \
--mask_dir ../translation/inputs/masks/day \
--origin_res

"""

import math
import argparse
import os
import glob
import time
import shutil

import numpy as np 
from skimage import img_as_float
from skimage.io import imread, imsave
from scipy.fftpack import dct, idct
from scipy.ndimage import correlate
from skimage.filters import gaussian, sobel_h, sobel_v, scharr_h, scharr_v, roberts_pos_diag, roberts_neg_diag, prewitt_h, prewitt_v
from skimage.transform import resize

import cv2

################## Gradient Operator #########################
normal_h = lambda im: correlate(im, np.asarray([[0, -1, 1]]), mode='nearest')
normal_v = lambda im: correlate(im, np.asarray([[0, -1, 1]]).T, mode='nearest')

gradient_operator = {
    'normal': (normal_h, normal_v), #default
    'sobel': (sobel_h, sobel_v),
    'scharr': (scharr_h, scharr_v),
    'roberts': (roberts_pos_diag, roberts_neg_diag),
    'prewitt': (prewitt_h, prewitt_v)
}


###########################################################

def ndarray_resize(im, image_size, order=3, dtype=None):
    im = resize(im, image_size, preserve_range=True, order=order, mode='constant')

    if dtype:
        im = im.astype(dtype)
    return im


def imfilter2d(im, filter_func):
    gradients = np.zeros_like(im)
    for i in range(im.shape[2]):
        gradients[:, :, i] = filter_func(im[:, :, i])

    return gradients


def gradient_feature(im, color_feature, gradient_kernel):
    result = np.zeros((*im.shape, 5))

    gradient_h, gradient_v = gradient_operator[gradient_kernel]

    result[:, :, :, 0] = color_feature
    result[:, :, :, 1] = imfilter2d(im, gradient_h)
    result[:, :, :, 2] = imfilter2d(im, gradient_v)
    result[:, :, :, 3] = np.roll(result[:, :, :, 1], 1, axis=1)
    result[:, :, :, 4] = np.roll(result[:, :, :, 2], 1, axis=0)

    return result.astype(im.dtype)


def fft2(K, size, dtype):
    w, h = size
    param = np.fft.fft2(K)
    param = np.real(param[0:w, 0:h])

    return param.astype(dtype)

# get laplacian filter (for image gradiant)
def laplacian_param(size, dtype):
    w, h = size
    K = np.zeros((2 * w, 2 * h)).astype(dtype)

    laplacian_k = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    kw, kh = laplacian_k.shape
    K[:kw, :kh] = laplacian_k

    K = np.roll(K, -(kw // 2), axis=0)
    K = np.roll(K, -(kh // 2), axis=1)

    return fft2(K, size, dtype)

# get gaussian
def gaussian_param(size, dtype):
    w, h = size
    K = np.zeros((2 * w, 2 * h)).astype(dtype)

    K[1, 1] = 1
    # K[:3, :3] = gaussian(K[:3, :3], sigma) # actually do not apply gaussian filtering

    K = np.roll(K, -1, axis=0)
    K = np.roll(K, -1, axis=1)

    return fft2(K, size, dtype)


def dct2(x, norm='ortho'):
    return dct(dct(x, norm=norm).T, norm=norm).T


def idct2(x, norm='ortho'):
    return idct(idct(x, norm=norm).T, norm=norm).T

# Different from GP-GAN, do not apply gaussian filtering 
def gp_editing(X, param_l, param_g, color_weight=1, eps=1e-12):
    Fh = (X[:, :, :, 1] + np.roll(X[:, :, :, 3], -1, axis=1)) / 2
    Fv = (X[:, :, :, 2] + np.roll(X[:, :, :, 4], -1, axis=0)) / 2
    L = np.roll(Fh, 1, axis=1) + np.roll(Fv, 1, axis=0) - Fh - Fv

    param = param_l + color_weight * param_g
    param[(param >= 0) & (param < eps)] = eps
    param[(param < 0) & (param > -eps)] = -eps

    Y = np.zeros(X.shape[:3])
    for i in range(3):
        Xdct = dct2(X[:, :, i, 0])
        Ydct = (dct2(L[:, :, i]) + color_weight * Xdct) / param
        Y[:, :, i] = idct2(Ydct)
    return Y


def run_editing(src, blended_im, mask_im, opt_im, color_weight, gradient_kernel='normal', whole_grad=False):
    # get geometry/gradient feature
    if whole_grad:
        bg_feature = gradient_feature(src, opt_im, gradient_kernel) # source background texture
    else:
        bg_feature = gradient_feature(blended_im, opt_im, gradient_kernel) # new background texture
    fg_feature = gradient_feature(src, opt_im, gradient_kernel) # foreground
    feature = bg_feature * (1 - mask_im) + fg_feature * mask_im # combined gradient feature

    # get parameters
    size, dtype = feature.shape[:2], feature.dtype
    param_l = laplacian_param(size, dtype) #gradient
    param_g = gaussian_param(size, dtype) #color

    # run editing
    opt_im = gp_editing(feature, param_l, param_g, color_weight=color_weight)
    opt_im = np.clip(opt_im, 0, 1)

    return opt_im

# style image pyramid in different sizes in log2 decreasing
def laplacian_pyramid(im, max_level, image_size):
    im_pyramid = [im]
    for i in range(max_level - 1, -1, -1):
        im_pyramid_last = ndarray_resize(im_pyramid[-1], (image_size * 2 ** i, image_size * 2 ** i))
        im_pyramid.append(im_pyramid_last)

    im_pyramid.reverse()
    return im_pyramid


"""
Image Blending Optimization

    src:  source image,      size: [H, W, 3], dtype: float, value: [0, 1]
    fg:  foreground image,      size: [H, W, 3], dtype: float, value: [0, 1]
    bg :  background image, size: [H, W, 3], dtype: float, value: [0, 1]
    mask: mask image,        size: [H, W],     dtype: float, value: [0, 1]
    
    image_size: short side size
    color_weight: weight for color constraint
    gradient_kernel: kernel type for calc gradient 

    n_iteration: # of iterations for optimization
"""
def blend_optimize(src, blended, mask, image_size=256, color_weight=1, gradient_kernel='normal',
                    n_iteration=1, whole_grad=False, origin_res=False):

    if origin_res:
        h_orig, w_orig, _ = src.shape
        blended = resize(blended, (h_orig, w_orig))
        mask = resize(mask, (h_orig, w_orig))
    else:
        h_orig, w_orig, _ = blended.shape
        if (h_orig > w_orig):
            if (h_orig < image_size):
                h_orig = image_size
                w_orig = int(w_orig * (image_size/h_orig) + 0.5)  
        elif (w_orig < image_size):
            w_orig = image_size
            h_orig = int(h_orig * (image_size/w_orig) + 0.5) 
    
        src = resize(src, (h_orig, w_orig)) 
        mask = resize(mask, (h_orig, w_orig))  
 
    ############################ Image Gaussian Poisson Editing #############################
    if n_iteration > 1: # in case for iterative optimization
        origin_color_weight = color_weight
        color_weight = 0.5
    
    final_grad = whole_grad
    for iter in range(n_iteration):
        if (n_iteration > 1) and (iter >= n_iteration-2):
            color_weight = origin_color_weight
            whole_grad = final_grad

        # pyramid
        max_level = int(math.ceil(np.log2(max(w_orig, h_orig) / image_size))) 
        blended_im_pyramid = laplacian_pyramid(blended, max_level, image_size)
        src_im_pyramid = laplacian_pyramid(src, max_level, image_size)

        # init image
        # mask_init = ndarray_resize(mask, (image_size, image_size), order=0)[:, :, np.newaxis] 
        # blended_init = fg_im_pyramid[0] * mask_init + bg_im_pyramid[0] * (1 - mask_init) 
        blended_init = blended_im_pyramid[0]
  
        opt_im = np.clip(blended_init, 0, 1).astype(blended.dtype)     
        # Start pyramid
        for level in range(max_level + 1):
            size = blended_im_pyramid[level].shape[:2]
            mask_im = ndarray_resize(mask, size, order=0)[:, :, np.newaxis, np.newaxis]
            if level != 0:
                opt_im = ndarray_resize(opt_im, size)
            opt_im = run_editing(src_im_pyramid[level], blended_im_pyramid[level], mask_im, opt_im, color_weight, gradient_kernel, whole_grad)
        if n_iteration > 1:
            src = opt_im

    opt_im = np.clip(opt_im * 255, 0, 255).astype(np.uint8)

    return opt_im


if __name__ == '__main__':

    start_time = time.time()

    parser = argparse.ArgumentParser()

    # Single image mode
    parser.add_argument('--single_image', action="store_true", help="enable to only blend one image")
    parser.add_argument('--source', '-s', type=str, default='src.jpg', help="image with perfect gradient.")
    parser.add_argument("--blended", '-b', type=str, default='blended.jpg', help="blended image") 
    parser.add_argument("--mask", '-m', type=str, default='mask.jpg', help="mask image for blending (foreground is white, background is black)")
    parser.add_argument('--opt_image', default='blended_opt.jpg', help='name of optimized image')
    
    parser.add_argument('--result_folder', default='opt_result', help='Name for folder storing results')

    # Folder mode
    parser.add_argument('--blended_dir', default='', help='blended image folder') 
    parser.add_argument('--src_dir', default='', help='whole image folder (w/ perfect gradient)')
    parser.add_argument('--mask_dir', default='', help='mask images with original resolution (fg:white, bg:black)')
    
    # Optimization parameters (default is used in paper)
    parser.add_argument('--image_size', type=int, default=256, help='Pyramid smallest image size.')
    parser.add_argument('--color_weight', type=float, default=1, help='Color weight, larger then lower weight for gradient restoration')
    parser.add_argument('--gradient_kernel', type=str, default='normal', help='Kernel type for calc gradient')
    parser.add_argument('--n_iteration', type=int, default=1, help='# (1 or 2) of iterations for optimizing gradient') 

    # If restore original high resolution
    parser.add_argument('--whole_grad', action="store_true", help="enable to apply gradient of whole source image, otherwise only retrieve foreground gradient")
    parser.add_argument('--origin_res', action="store_true", default=False, help="Keep original high resolution")

    args = parser.parse_args()

    if not os.path.isdir(args.result_folder):
        os.makedirs(args.result_folder)
    print('Result will save to {} ...\n'.format(args.result_folder))

    if args.single_image:
        # load images
        src = img_as_float(imread(args.source)) 
        blended = img_as_float(imread(args.blended))
        mask = imread(args.mask, as_gray=True).astype(blended.dtype)
        
        # optimize
        opt_im = blend_optimize(src, blended, mask, args.image_size, color_weight=args.color_weight,
                                gradient_kernel=args.gradient_kernel, n_iteration=args.n_iteration, whole_grad=args.whole_grad)
        imsave(os.path.join(args.result_folder, args.opt_image), opt_im)

        print("--- Time elapsed: %.4f sec/minutes ---" %((time.time() - start_time)/60))
    else:
        src_dir = args.src_dir 
        blended_dir = args.blended_dir
        mask_dir = args.mask_dir  
        result_folder = args.result_folder

        blended_folders = sorted(os.listdir(blended_dir))
        src_images = sorted(glob.glob(os.path.join(src_dir,"*")))
        mask_images= sorted(glob.glob(os.path.join(mask_dir,"*")))    
        
        # it is based on default output format
        '''
          idx1: folder index (source image index)
          idx2: output image index
        '''
        for cnt, idx in enumerate(blended_folders):
                try:
                    idx1 = int(idx)
                except:
                    continue # if idx is not an index folder

                print("%d/%d, folder %d" %(cnt+1, len(blended_folders), int(idx)))
                # folder idx, for one scene
                blended_images = glob.glob(os.path.join(blended_dir, str(idx1), "output_*_blend.jpg")) # read images 
                
                output_img_folder = os.path.join(result_folder, str(idx1))
                if not os.path.exists(output_img_folder):
                    os.makedirs(output_img_folder)

                # load source image and its mask
                src = img_as_float(imread(src_images[idx1])) 
                mask = imread(mask_images[idx1], as_gray=True).astype(src.dtype) 
                 
                if not args.origin_res:
                    src = resize(src, blended.shape)
                shutil.copyfile(src_images[idx1], os.path.join(output_img_folder, "input.jpg")) # save input daytime image


                # optimized images for the same source/each folder 
                blended_img_dir = os.path.join(blended_dir, str(idx1)) 
                for idx2 in range(len(blended_images)):
                    filename = os.path.basename(blended_images[idx2])
                    blended = img_as_float(imread(os.path.join(blended_img_dir, filename))) # read images  

                    # optimize image
                    opt_im = blend_optimize(src, blended, mask, args.image_size, color_weight=args.color_weight,
                                            gradient_kernel=args.gradient_kernel, n_iteration=args.n_iteration, whole_grad=args.whole_grad, origin_res=args.origin_res) 
                    # save image
                    filename = filename.split('.')[0] + ".png"
                    imsave(os.path.join(output_img_folder, filename ), opt_im) 


        print("--- Time elapsed: %.4f minutes ---" %((time.time() - start_time)/60))
