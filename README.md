# Time-of-Day Neural Style Transfer for Architectural Photographs


<!-- <a href="https://chenyingshu.github.io/time_of_day/"><img src="https://img.shields.io/badge/WEBSITE-Visit%20project%20page-blue?style=for-the-badge"></a> -->
<!-- <a href=""><img src="https://img.shields.io/badge/arxiv-2112.00719-red?style=for-the-badge"></a> -->

[Yingshu Chen]()<sup>1</sup>,
[Tuan-Anh Vu]()<sup>1</sup>,
[Ka-Chun Shum]()<sup>1</sup>,
[Binh-Son Hua](https://sonhua.github.io/)<sup>2</sup>,
[Sai-Kit Yeung](https://www.saikit.org/)<sup>1</sup> <br>
<sup>1</sup>The Hong Kong University of Science and Technology, <sup>2</sup> VinAI Research

Code release for ICCP 2022 paper "Time-of-Day Neural Style Transfer for Architectural Photographs".

> **Abstract:** 
Architectural photography is a genre of photography that focuses on capturing a building or structure in the foreground with dramatic lighting in the background. Inspired by recent successes in image-to-image translation methods, we aim to perform style transfer for architectural photographs. However, the special composition in architectural photography poses great challenges for style transfer in this type of photographs. Existing neural style transfer methods treat the architectural images as a single entity, which would generate mismatched chrominance and destroy geometric features of the original architecture, yielding unrealistic lighting, wrong color rendition, and visual artifacts such as ghosting, appearance distortion, or color mismatching. In this paper, we specialize a neural style transfer method for architectural photography. Our method addresses the composition of the foreground and background in an architectural photograph in a two-branch neural network that separately considers the style transfer of the foreground and the background, respectively. Our method comprises a segmentation module, a learning-based image-to-image translation module, and an image blending optimization module. We trained our image-to-image translation neural network with a new dataset of unconstrained outdoor architectural photographs captured at different magic times of a day, utilizing additional semantic information for better chrominance matching and geometry preservation. Our experiments show that our method can produce photorealistic lighting and color rendition on both the foreground and background, and outperforms general image-to-image translation and arbitrary style transfer baselines quantitatively and qualitatively. Our code and data will be made available to facilitate future work on this problem.

## Get Started
Source code will be released soon.

## Dataset
The released dataset is for non-commercial use only.

- **Training set:**
Training set will be released soon.
A request is needed for training data access. 

- **Evaluation set:**
The evaluation set contains 1,003 images in four time styles. <br>
Evaluation set used in the paper: [Download Link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ychengw_connect_ust_hk/ERdVPaeZXgBNo0rluxa9qBwBSufzDo0y1Gy2bRRPNYNOPQ?e=aEtKPU). <br>
If you want to get evaluation images in original high resolution with source information, please download data here: [Download Link](https://hkustconnect-my.sharepoint.com/:u:/g/personal/ychengw_connect_ust_hk/ERZUW4-GmPtNm3C2OacU_Y8BAVrMWah3cW5kJwvkvbbGKw?e=ZcnqgD).

## Citation
If you find our data or work useful in your research, please consider citing: 
```bibtex
@inproceedings{chen2022timeofday,
  title={Time-of-Day Neural Style Transfer for Architectural Photographs},
  author={Chen, Yingshu and Vu, Tuan-Anh and Shum, Ka-Chun and Hua, Binh-Son and Yeung, Sai-Kit},
  booktitle={International Conference on Computational Photography (ICCP)},
  year={2022},
  organization={IEEE}
}
```
