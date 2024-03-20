#!/usr/bin/env bash
# command to install this enviroment: source install.sh

# install PyTorch
conda deactivate
conda env remove --name 2.5d-segmentation
conda create -n 2.5d-segmentation
conda activate 2.5d-segmentation
# #NOTE: Please note cuda version. Mine is 12.0. Either update nvidia driver or select the appropriate cuda version.
conda install -y python=3.9.7

# install relevant packages
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install timm
pip install tensorboardX
pip install torchmetrics
pip install matplotlib
pip install scikit-learn
pip install pandas
pip install opencv-python
pip install kornia
pip install open3d
pip install pymeshfix
pip install dropbox
pip install -U coremltools