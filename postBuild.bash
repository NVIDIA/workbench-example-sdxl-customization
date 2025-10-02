#!/bin/bash
# This file contains bash commands that will be executed at the end of the container build process,
# after all system packages and programming language specific package have been installed.
#
# Note: This file may be removed if you don't need to use it

cd /workspace
git clone https://github.com/huggingface/diffusers && \
    cd diffusers && \
    pip install -e .
cd -

sudo pip uninstall torch torchvision torchaudio -y
sudo pip install --pre torch torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130

sudo mkdir -p /mnt/cache/
sudo chown $NVWB_UID:$NVWB_GID /mnt/cache/