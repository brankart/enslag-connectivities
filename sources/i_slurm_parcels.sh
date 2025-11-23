#!/bin/bash
#SBATCH --account=c1601279
#SBATCH --job-name="runseq_parcels_swiose"
#SBATCH -e jobs_i/parcels_swiose.error%j
#SBATCH -o jobs_i/parcels_swiose.output%j
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --exclusive                              # Shared !
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1                          # 16 threads or 16/2 cores.
#SBATCH --threads-per-core=1
#SBATCH --time=0:30:00



module purge
module load anaconda3/2023.09-0
module load ncview/2.1.9-mpi
module list

conda activate parcels


salloc --account=c1601279 --constraint=GENOA --job-name="runseq_parcels_swiose" --nodes=1 --time=03:00:00 --exclusive
