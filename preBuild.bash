#!/bin/bash
# This file contains bash commands that will be executed at the beginning of the container build process,
# before any system packages or programming language specific package have been installed.
#
# Note: This file may be removed if you don't need to use it

# Get the machine architecture
ARCH=$(uname -m)

# If ARM architecture, user is on DGX Spark
if [[ "$ARCH" == "arm"* || "$ARCH" == "aarch64" ]]; then
    echo "Detected ARM architecture. User is on Spark; installing CUDA 13.0"
    cd /tmp
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-ubuntu2204.pin
    sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda-repo-ubuntu2204-13-0-local_13.0.1-580.82.07-1_arm64.deb
    sudo dpkg -i cuda-repo-ubuntu2204-13-0-local_13.0.1-580.82.07-1_arm64.deb
    sudo cp /var/cuda-repo-ubuntu2204-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-13-0
else
    echo "Architecture is not ARM. Skipping CUDA 13.0"
fi

# Update base container image to CUDA 13.0 (DGX Spark)

