# Multiresolution Compressive Feature Fusion for Spectral Image Classification

[Juan Marcos Ramirez](juanra@ula.ve) and [Henry Arguello Fuentes](henarfu@uis.edu.co)

## Abstract
Compressive spectral imaging (CSI) has emerged as an alternative acquisition framework that senses and compresses spectral images simultaneously. In this context, spectral image classification from CSI compressive measurements has become a challenging task since the feature extraction stage usually requires to reconstruct the spectral image. Moreover, most of these approaches do not consider multi-sensor compressive measurements. In this paper, an approach for fusing features obtained from multi-sensor compressive measurements is proposed for spectral image classification. To this end, linear models describing low-resolution features as degraded versions of the high-resolution features are developed. Furthermore, an inverse problem is formulated aiming at estimating high-resolution features including both a sparsity-inducing term and a total variation (TV) regularization term to exploit the correlation between neighboring pixels of the spectral image, and therefore, to improve the performance of pixel-based classifiers. An algorithm based on the alternating direction method of multipliers (ADMM) is described for solving the fusion problem. The proposed feature fusion approach is tested for two CSI architectures: 3D-CASSI and C-CASSI. Extensive simulations on various spectral image datasets show that the proposed approach outperforms other classification approaches under different performance criteria. 

## Suplementary Material

Reproducible research: Multiresolution Compressive Feature Fusion for Spectral Image Classification

### How to run the code

Download and uncompress the `FeatureFusionHDSP_UIS` folder. Under **MATLAB** environment, navigate to the `FeatureFusionHDSP_UIS` folder and follow the instructions described below

#### Figure 3

To generate Figures 3, in MATLAB, 

	>> figure3_results.m
 	
#### Table I

	>> table1_results.m

### Platform

* CentOS 7 Operating System, MATLAB R2017b

## License

This code package is licensed under the GNU GENERAL PUBLIC LICENSE (version 3)- see the [LICENSE](LICENSE) file for details

## Author

Juan Marcos Ramírez Rondón

### Contact

[Juan Marcos Ramirez](juanra@ula.ve)

## Date

March 22, 2019
