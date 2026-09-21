#!/bin/bash
#SBATCH --account=c1601279
#SBATCH --job-name="runseq_parcels_swiose"
#SBATCH -e jobs/parcels_swiose.error%j
#SBATCH -o jobs/parcels_swiose.output%j
#SBATCH --constraint=GENOA
#SBATCH --nodes=1
#SBATCH --exclusive                              # Shared !
#SBATCH --hint=nomultithread
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1                          # 16 threads or 16/2 cores.
#SBATCH --threads-per-core=1
#SBATCH --time=03:00:00

module purge
module load cpe/24.07
module load craype-x86-genoa
module load PrgEnv-cray
module load anaconda3/2023.09-0
module load udunits/2.2.28
module list
conda activate parcels

echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "Current working directory: $(pwd)"

# srun python $SLURM_SUBMIT_DIR/runseq_swiose.py
srun python /lus/scratch/CT1/c1601279/tbellayer/parcels/tests_FULLENSswiose/runseq_swiose.py
# srun python /lus/scratch/CT1/c1601279/tbellayer/parcels/tests_FULLENSswiose/getvars_swiose.py

