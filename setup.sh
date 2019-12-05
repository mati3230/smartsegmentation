#!/bin/bash

sudo apt-get install git

DIR="segmentation"
if [ ! -d ${DIR} ]
then
    git clone https://github.com/mati3230/segmentation.git
fi

cd ${DIR}
sh setup.sh
cd ..

sudo apt-get install openmpi-bin libopenmpi-dev

DIR="stable-baselines"
if [ ! -d ${DIR} ]
then
    git clone https://github.com/mati3230/stable-baselines.git
fi
cd ${DIR}
pip install -e .
cd ..


TF_VERSION="-gpu==1.14"
if [ -n "$1" ]; then # if first parameter passed
    if [ $1 = "tensorflow==1.5" ] 
    then
        TF_VERSION="==1.5"
        
    elif [ $1 = "tensorflow==1.14" ] 
    then
        TF_VERSION="==1.14"
    fi
fi
echo "tensorflow${TF_VERSION} will be installed" 
pip install tensorflow${TF_VERSION}
pip install -r requirements.txt
