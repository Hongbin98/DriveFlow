# 🌠 DriveFlow
This is the official repository for the AAAI 2026 paper "*[DriveFlow: Rectified Flow Adaptation for Robust 3D Object Detection in Autonomous Driving](https://arxiv.org/abs/2511.18713)*".

## Abstract
In autonomous driving, vision-centric 3D object detection recognizes and localizes 3D objects from RGB images. However, due to high annotation costs and diverse outdoor scenes, training data often fails to cover all possible test scenarios, known as the out-of-distribution (OOD) issue. Training-free image editing offers a promising solution for improving model robustness by training data enhancement without any modifications to pre-trained diffusion models. Nevertheless, inversion-based methods often suffer from limited effectiveness and inherent inaccuracies, while recent rectified-flow-based approaches struggle to preserve objects with accurate 3D geometry. In this paper, we propose DriveFlow, a Rectified Flow Adaptation method for training data enhancement in autonomous driving based on pre-trained Text-to-Image flow models. Based on frequency decomposition, DriveFlow introduces two strategies to adapt noise-free editing paths derived from text-conditioned velocities. 1) High-Frequency Foreground Preservation: DriveFlow incorporates a high-frequency alignment loss for foreground to maintain precise 3D object geometry. 2) Dual-Frequency Background Optimization: DriveFlow also conducts dual-frequency optimization for background, balancing editing flexibility and semantic consistency. Comprehensive experiments validate the effectiveness and efficiency of DriveFlow, demonstrating comprehensive performance improvements on all categories across OOD scenarios.

## Demo
https://github.com/user-attachments/assets/38daed89-8b98-453e-abbb-743bed17ac2a

## Citation
If our DriveFlow method is helpful in your research, please consider citing our paper:
```
@article{lin2025driveflow,
  title={DriveFlow: Rectified Flow Adaptation for Robust 3D Object Detection in Autonomous Driving},
  author={Lin, Hongbin and Yang, Yiming and Zheng, Chaoda and Zhang, Yifan and Niu, Shuaicheng and Guo, Zilu and Li, Yafeng and Gui, Gui and Cui, Shuguang and Li, Zhen},
  journal={arXiv preprint arXiv:2511.18713},
  year={2025}
}
```

## Acknowledgment
The code is greatly inspired by (heavily from) the [FlowEdit🔗](https://github.com/fallenshock/FlowEdit).

## Correspondence 
Please contact Hongbin Lin by [linhongbinanthem@gmail.com] if you have any questions.  📬


