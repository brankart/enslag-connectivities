#!/bin/bash
#SBATCH --account={your_account}
#SBATCH --job-name="runparcels_ini"
#SBATCH -e ../jobs/parcels_swiose.error%j
#SBATCH -o ../jobs/parcels_swiose.output%j
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --exclusive # exclusie or shared mode                            
#SBATCH --time=00:10:00

module purge
module load cpe/24.07
module load craype-x86-genoa
module load PrgEnv-gnu
module load cray-mpich
module load cray-python
module list

# Module diags
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
ldd ${WORKDIR}/parcels/lib/python3.*/site-packages/mpi4py/*.so

# !!! IF RUN WITH CRAYPE (RECOMMENDED ON ADASTRA) :
source ${WORKDIR}/parcels/bin/activate

# Diags
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "Current working directory: $(pwd)"

# srun --ntasks=1 python /lus/scratch/CT1/c1601279/tbellayer/connectivities/src/runpar_ini_opt.py

srun --ntasks-per-node={} --cpus-per-task=1-{} --threads-per-core={} -- ${WORKDIR}/parcels/bin/python3 runpar_ini_opt.py

