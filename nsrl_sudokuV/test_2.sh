#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=112G
#SBATCH --time=50:00:00
export OMP_NUM_THREADS=1

source activate singleRL

python PPO_discrete_main.py --max_train_steps=45000 --algorithm_name=ppo --env_name=sudoku3