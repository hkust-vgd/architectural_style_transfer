# Time-of-Day Neural Style Transfer for Architectural Photographs


<a href="https://chenyingshu.github.io/time_of_day/"><img src="https://img.shields.io/badge/WEBSITE-Visit%20project%20page-blue?style=for-the-badge"></a>
<!-- <a href="https://github.com/hkust-vgd/architectural_style_transfer"><img src="https://img.shields.io/badge/CODE-Access%20Github-red?style=for-the-badge"></a> -->

[Yingshu Chen]()<sup>1</sup>,
[Tuan-Anh Vu]()<sup>1</sup>,
[Ka-Chun Shum]()<sup>1</sup>,
[Binh-Son Hua](https://sonhua.github.io/)<sup>2</sup>,
[Sai-Kit Yeung](https://www.saikit.org/)<sup>1</sup> <br>
<sup>1</sup>The Hong Kong University of Science and Technology, <sup>2</sup> VinAI Research

> **Abstract:** 
Architectural photography is a genre of photography that focuses on capturing a building or structure in the foreground with dramatic lighting in the background. Inspired by recent successes in image-to-image translation methods, we aim to perform style transfer for architectural photographs. However, the special composition in architectural photography poses great challenges for style transfer in this type of photographs. Existing neural style transfer methods treat the architectural images as a single entity, which would generate mismatched chrominance and destroy geometric features of the original architecture, yielding unrealistic lighting, wrong color rendition, and visual artifacts such as ghosting, appearance distortion, or color mismatching. In this paper, we specialize a neural style transfer method for architectural photography. Our method addresses the composition of the foreground and background in an architectural photograph in a two-branch neural network that separately considers the style transfer of the foreground and the background, respectively. Our method comprises a segmentation module, a learning-based image-to-image translation module, and an image blending optimization module. We trained our image-to-image translation neural network with a new dataset of unconstrained outdoor architectural photographs captured at different magic times of a day, utilizing additional semantic information for better chrominance matching and geometry preservation. Our experiments show that our method can produce photorealistic lighting and color rendition on both the foreground and background, and outperforms general image-to-image translation and arbitrary style transfer baselines quantitatively and qualitatively. 

## Get Started
<!--:eyes: Source code will be released soon. Please stay tuned.:eyes:-->

### Requirements
<!-- Tested with Python 3.6 or above + Pytorch 1.6 + GTX 1080 Ti with 11GB memory (CUDA 10.1). <br> -->
Tested with:
- Python 3.6 or above
- [Pytorch](https://pytorch.org/) 1.7 or above
- GTX 2080 Ti with 11GB memory (CUDA 10.2) or GTX 3090 Ti with 24GB memory (CUDA 11.0)

Others:
- tensorboard, tensorboardX
- pyyaml
- pillow
- scikit-image

### Quick Start 
1. Clone github repository:
```
git clone https://github.com/hkust-vgd/architectural_style_transfer.git
cd architectural_style_transfer/translation
```
2. Download [pretrained models](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ychengw_connect_ust_hk/EfrezLEVWgZCtqCbAD_2d9YBAtz722sxbMfxXXSJmPK2tA?download=1) (trained with 256x256 images), and put them under folder `translation/checkpoints`:
```
bash checkpoints/download_models.sh
```
3. Run testing script:
```
bash test_script.sh
```

### Data Segmentation Processing
Segmentation map contains only two labels, white color for foreground, black color for background (i.e., sky). See samples in `translation/inputs/masks` and `translation/training_samples/masks`.

#### Manual labeling
You can manually label sky as background, remaining as foreground. <br>
At testing, manual labeling for input source image is recommended for better blended results.

#### Automatic labeling
We used pretrained model (`ResNet50dilated + PPM_deepsup`) to label sky background for training and evaluation data as described in the paper.
Please access this [repository](https://github.com/CSAILVision/semantic-segmentation-pytorch#supported-models) for details.

### Testing

#### Style transfer
1. Download and put [pretrained models](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ychengw_connect_ust_hk/EfrezLEVWgZCtqCbAD_2d9YBAtz722sxbMfxXXSJmPK2tA?download=1) (trained with 256x256 images) in `translation/checkpoints`.
2. Prepare a set of daytime images and a set of target style images in same domain (e.g., golden style), and put them in `TEST_ROOT/day` and `TEST_ROOT/TARGET_CLASS`.
3. Prepare segmentation maps (white for foreground, black for background) for all images, and put them in `MASK_ROOT/day` and `MASK_ROOT/TARGET_CLASS`.
4. Decide inference image size, e.g., NEW_SIZE = 256x, 512x or 1024x resolution. Multiple of $2^5$ is recommended.
5. Run testing script (see *test_script.sh* as an example):
```
CUDA_VISIBLE_DEVICES=1 python test.py \
--test_root TEST_ROOT_DIR \
--mask_root MASK_ROOT_DIR \
-a day \
-b TARGET_CLASS \
--output_path results \
--config_fg checkpoints/config_day2golden_fg.yaml \
--config_bg checkpoints/config_day2golden_bg.yaml \
--checkpoint_fg checkpoints/gen_day2golden_fg.pt \
--checkpoint_bg checkpoints/gen_day2golden_bg.pt \
--new_size NEW_SIZE \
--opt
```
You can view results in html by running:
```
python gen_html.py -i ./results 
```

<!--
#### Style interpolation
1. Download [pretrained models](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ychengw_connect_ust_hk/EfrezLEVWgZCtqCbAD_2d9YBAtz722sxbMfxXXSJmPK2tA?download=1) (trained with 256x256 images).
2. Prepare a daytime image and two target style images of same class (each in any resolution).
3. Prepare segmentation maps (white for foreground, black for background) for all images.
4. Run testing script:
```
TBD
```
-->

### Training
Training is tested in NVIDIA GeForce RTX 2080 Ti with 11GB memory with one single GPU under 256x256 resolution,
and in NVIDIA GeForce RTX 3090 Ti with 24GB memory with one single GPU under 512x512 resolution (batch=1).

1. Download training data in [Dataset](#dataset).
2. Select source data and target style data for training, e.g., `day` and `golden`.
3. Preprocess data with foreground and background segmentation (assume you finish labeling, see details in [Data Segmentation Processing](#data-segmentation-processing)):
```
python mask_images.py \
--img_dir training_samples/CLASS_NAME \
--mask_dir training_samples/masks/CLASS_NAME \
--class_name CLASS_NAME \
--output_dir training_samples \
--kernel_size 0
```
3. Configure training parameters (we trained models in cropped 256x256 of resized 286x286 images) and save configuration file as `translation/configs/XXXX.yaml`.
4. Run training script, for example:
```
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/day2golden_fg.yaml  \
--save_name day2golden_fg
```

## Blending Optimization
You can run blending optimization solely after image translation based on translated results, for example:
```
cd optimization
python blend_opt.py \
--blended_dir ../translation/results/ \
--src_dir ../translation/inputs/images/day/ \
--mask_dir ../translation/inputs/masks/day \
--origin_res
```

## Dataset
The Time-lapse Architectural Style Transfer dataset is released for :warning:**non-commercial**:warning: use only.

The dataset is manually classified into four classes of time-of-day styles: `day`,`golden`,`blue`, `night`.

- **Training set:**
Training set will be released soon.
A [request form](https://forms.gle/wUrXgdWAEki73B9X9) is required to be filled for training data access (7.4GB).

- **Evaluation set:**
The evaluation set contains 1,003 images in four time styles. <br>
Evaluation set used in the paper: [Download Link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ychengw_connect_ust_hk/ERdVPaeZXgBNo0rluxa9qBwBSufzDo0y1Gy2bRRPNYNOPQ?download=1) (550MB). <br>
If you want to get evaluation images in original high resolution with source information, please download data here: [Download Link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ychengw_connect_ust_hk/ERZUW4-GmPtNm3C2OacU_Y8BAVrMWah3cW5kJwvkvbbGKw?download=1) (2.2GB). Please check image original sources for other usages (e.g., commercial use).

- **Segmentation maps:**
Please refer to [Data Segmentation Processing](#data-segmentation-processing) for data processing details. <br>
<!--You can also download manual labelled testing and evaluation segmentation maps: [TBD]().-->

## Citation
If you find our work or data useful in your research, please consider citing: 
```bibtex
@inproceedings{chen2022timeofday,
  title={Time-of-Day Neural Style Transfer for Architectural Photographs},
  author={Chen, Yingshu and Vu, Tuan-Anh and Shum, Ka-Chun and Hua, Binh-Son and Yeung, Sai-Kit},
  booktitle={International Conference on Computational Photography (ICCP)},
  year={2022},
  organization={IEEE}
}
```
## Contacts
Github issues are welcomed. You can also drop an email to yingshu2008[AT]gmail[DOT]com.

## Liscense
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

# Acknowledgements
The code borrows from [MUNIT](https://github.com/NVlabs/MUNIT), [DSMAP](https://github.com/acht7111020/DSMAP), and [GP-GAN](https://github.com/wuhuikai/GP-GAN).
