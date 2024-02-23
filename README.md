# DiffTransBEV

![image](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)  ![image](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

Generating optimized noise-free BEV representaiton of Autonomous Vehicles (Self-driving cars) using Stable Diffusion Transformers (DiTs), Diffusion Models, SwinV2 Transformers and Lift-Splat-Shoot.

<img src="visualizations/nuscenes_1.jpg" width=33% height=33%> <img src="visualizations/nuscenes_2.jpg" width=33% height=33%> <img src="visualizations/nuscenes_3.jpg" width=33% height=33%>

![CARLA Simulator Visualization](visualizations/carla_sim.jpg)

## Installation

This project works best on:  
* Python 3.7

### Steps to recreate the environment

1. Create conda environment for the project
```
conda create --name av-bev python=3.7
conda activate av-bev
```
2. Install PyTorch  
```
conda install pytorch torchvision -c pytorch
```
