# Point Cloud Segmentation with Deep Reinforcement Learning

A documentation of the code and the parameters will be provided soon. Furthermore, windows 10 installation instructions will also be provided soon.

## Requirements

A python interpreter with version 3.6 is assumed.

* [Segmentation Environment](https://github.com/mati3230/segmentation)
* [Stable-Baselines](https://github.com/mati3230/stable-baselines)
* [Pyntcloud](https://github.com/mati3230/pyntcloud)
* numpy
* scipy
* pandas
* matplotlib
* gym
* tensorflow-gpu==1.14 (tensorflow==1.14 without the usage of the GPU can also be used)

## Installation

The code is tested on Ubuntu 18.04.3.

1. Clone this repository
2. cd smartsegmentation
3. sh setup.sh

## Training

*python train.py*

The different parameters can be printed with "python train.py -h". By default the PointNet policy will be used.
To use the LDGCNN policy enter:

*python train.py --policy=ldgcnn*

To use the voxel based policy enter:

*python train.py --policy=vox_custom --point_mode=Voxel*

The point cloud scenes should be placed in a folder called *PointcloudScenes*. The scenes have the naming convention *scene_x.csv* whereas x is a natural number. Hence, the first scene is *scene_0.csv*, the second scene *scene_1.csv* and so on. The scenes will be loaded in ascending order. The parameter *max_scenes* determine how many scenes will be loaded.

## Manual Segmentation

The segmentation can be executed manually with:

*python manual_play.py*

Every step, the sampled observation will be plotted. The query points will be plotted in grey. The parameters can be entered as in the following example to segment the ceiling: 

* Seed Point X: 0
* Seed Point Y: 0
* Seed Point Z: 3
* K: 12
* Angle Threshold: 20
* Curvature Threshold: 0.1

## Testing/Agent Segmentation

The segmentation by the agent and the result will be plotted. A trained policy is necessary. Policies are stored by default in the *./save_model/* directory. This directory can be changed with the *checkpoint_dir* parameter. 

*python agent_play.py*

Similar to the training process, the policies can be changed with:

*python agent_play.py --policy=ldgcnn* or *python agent_play.py --policy=vox_custom --point_mode=Voxel*

## Plot Point Clouds

To plot a point cloud use: 

*python plot.py --file=PointcloudScenes/scene_0.csv*

To see the different labels enter: 

*python plot.py --file=PointcloudScenes/scene_0.csv --color_labels=True*

To plot the curvature values type: 

*python plot.py --file=PointcloudScenes/scene_0.csv --color_labels=True --curvature=True*

Moreover the point cloud with the curvature values will be plotted. The points with high curvature will be plotted in red. 