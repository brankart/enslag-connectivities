#!/bin/bash

set -eu

module purge
module load cpe/24.07
# module load craype-x86-genoa
module load craype-x86-rome # CPU-only MPI
module load PrgEnv-gnu
module load cray-mpich
module load cray-python
module list

source ${WORKDIR}/parcels/bin/activate

export MPI4PY_BUILD_MPICC="cc"  # Use Cray wrapper

pip3 install Cython
pip3 install --verbose --no-binary=mpi4py --compile --no-cache-dir --no-build-isolation mpi4py

echo "cray-MPI successfully built in parcels venv."


