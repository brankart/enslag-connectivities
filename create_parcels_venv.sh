#!/bin/bash

# Adastra environment
module purge
module load cpe/24.07
module load craype-x86-genoa
module load PrgEnv-gnu
module load cray-python
module list

# Install or upgrade the env management packages
python3 -m pip install --user --upgrade pip
pip3 install --user --upgrade virtualenv

export ENV_DIR="/lus/scratch/CT1/c1601279/SHARED/debug_parcels"

# Create and activate a dedicated environment
python3 -m virtualenv ${ENV_DIR}/parcels
chmod +x ${ENV_DIR}/parcels/bin/activate
source ${ENV_DIR}/parcels/bin/activate
python3 -m pip install --upgrade pip

# Install the needed packages
pip3 install parcels

