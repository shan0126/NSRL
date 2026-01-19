#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=112G
#SBATCH --time=50:00:00
export OMP_NUM_THREADS=1

source activate singleRL

python logicEncoder.py 