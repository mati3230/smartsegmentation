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


tf_version="-gpu==1.14"
if [ -n "$1" ]; then # if first parameter passed
    if [ $1 = "tensorflow==1.5" ] 
    then
        tf_version="==1.5"
    elif [ $1 = "tensorflow==1.14" ] 
    then
        tf_version="==1.14"
    fi
fi
echo "tensorflow$tf_version will be installed" 

pip install -r requirements.txt

git clone https://github.com/mati3230/pyntcloud.git
cd pyntcloud
pip install -e .
cd ..
