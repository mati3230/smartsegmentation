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
* tensorflow-gpu==1.14

## Installation

The code is tested on Ubuntu 18.04.3.

1. Clone this repository
2. cd smartsegmentation
3. sh setup.sh

## Training

python train.py

The different parameters can be printed with "python train.py -h". Have a look at the file "config.py" to see the default parameters.
By default the PointNet policy will be used.
To use the LDGCNN policy enter:

python train.py --policy=ldgcnn

To use the voxel based policy enter:

python train.py --policy=vox_custom --point_mode=Voxel

## Manual Segmentation

python manual_play.py

## Testing/Agent Segmentation

python agent_play.py