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

pip install peft==0.9.0 huggingface_hub[cli,torch]==0.21.4

sudo mkdir -p /mnt/cache/
sudo chown workbench:workbench /mnt/cache/
