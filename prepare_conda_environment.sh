#!/usr/bin/env bash

# Create env and install dependencies
conda install -c pytorch pytorch torchvision matplotlib

# Install SMAC and QPLEX SMAC MATRIX GAMES
git clone https://github.com/oxwhirl/smac && cd smac && git checkout tags/v1 
cd ..
