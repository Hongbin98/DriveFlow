# 🌠 DriveFlow
This is the official repository for the AAAI 2026 paper "*[DriveFlow: Rectified Flow Adaptation for Robust 3D Object Detection in Autonomous Driving](https://arxiv.org/abs/2511.18713)*".

## Abstract
In autonomous driving, vision-centric 3D object detection recognizes and localizes 3D objects from RGB images. However, due to high annotation costs and diverse outdoor scenes, training data often fails to cover all possible test scenarios, known as the out-of-distribution (OOD) issue. Training-free image editing offers a promising solution for improving model robustness by training data enhancement without any modifications to pre-trained diffusion models. Nevertheless, inversion-based methods often suffer from limited effectiveness and inherent inaccuracies, while recent rectified-flow-based approaches struggle to preserve objects with accurate 3D geometry. In this paper, we propose DriveFlow, a Rectified Flow Adaptation method for training data enhancement in autonomous driving based on pre-trained Text-to-Image flow models. Based on frequency decomposition, DriveFlow introduces two strategies to adapt noise-free editing paths derived from text-conditioned velocities. 1) High-Frequency Foreground Preservation: DriveFlow incorporates a high-frequency alignment loss for foreground to maintain precise 3D object geometry. 2) Dual-Frequency Background Optimization: DriveFlow also conducts dual-frequency optimization for background, balancing editing flexibility and semantic consistency. Comprehensive experiments validate the effectiveness and efficiency of DriveFlow, demonstrating comprehensive performance improvements across all categories in OOD scenarios.

## Demo
https://github.com/user-attachments/assets/38daed89-8b98-453e-abbb-743bed17ac2a


## Data Preparation
We follow [DriveGEN](https://github.com/Hongbin98/DriveGEN) to prepare the datasets.

### Monocular 3D object detection
- 1️⃣ Download the KITTI dataset from the *[official website](https://www.cvlibs.net/datasets/kitti/)*
- 2️⃣ Download the splits (the ImageSets folder) from *[MonoTTA](https://github.com/Hongbin98/MonoTTA/tree/main/ImageSets)*

Then, link the data by
```
mv ./ImageSets ./your_path_KITTI
mkdir data && cd data
ln -s /your_path_KITTI ./data/
```

## Installation
Build the conda environment
```
conda create -n driveflow python=3.10 -y
conda activate driveflow
# install pytorch like 'pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121'
pip install -r requirements.txt
```

## Usage
### Monocular 3D object detection
```
#### stable-diffusion-3-medium
python KITTI_driveflow.py    
```
And you will see the output images within the snowy scene, taking just several seconds (~5s/img on a single A100)!

### Enhance existing methods
Now you can leverage these generated images to enhance the training process of existing Monocular 3D Object Detectors (e.g., MonoFlex, MonoGround, MonoCD) since we reuse the initial object annotations.

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
