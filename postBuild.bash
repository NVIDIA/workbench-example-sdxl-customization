#!/bin/bash
# This file contains bash commands that will be executed at the end of the container build process,
# after all system packages and programming language specific package have been installed.
#
# Note: This file may be removed if you don't need to use it

# Get the machine architecture
ARCH=$(uname -m)

cd /workspace
git clone https://github.com/huggingface/diffusers && \
    cd diffusers && \
    git checkout 0974b4c6067165434fa715654b355b41beb5fceb && \
    pip install -e .
cd -

# Match diffusers package version
sudo pip install huggingface-hub==0.34.0

if [[ "$ARCH" == "arm"* || "$ARCH" == "aarch64" ]]; then
    # If ARM architecture, user is on DGX Spark
    echo "Detected ARM architecture. User is on Spark; installing CUDA 13.0 compatible torch"
    sudo pip uninstall torch torchvision torchaudio -y
    sudo pip install --pre torch torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130
else
    # Otherwise, install normal torch
    echo "Architecture is not ARM. Installing standard torch"
    sudo pip install torch==2.6.0 torchvision==0.21.0
fi

sudo mkdir -p /mnt/cache/
sudo chown $NVWB_UID:$NVWB_GID /mnt/cache/