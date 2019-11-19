#!/bin/bash

git clone https://github.com/mati3230/segmentation.git
cd segmentation
sh setup.sh
cd ..

sudo apt-get install openmpi-bin libopenmpi-dev
git clone https://github.com/mati3230/stable-baselines.git
cd stable-baselines
pip install -e .
cd ..

git clone https://github.com/mati3230/pyntcloud.git
cd pyntcloud
pip install -e .
cd ..

pip install -r requirements.txt
