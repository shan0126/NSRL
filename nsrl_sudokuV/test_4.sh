#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=112G
#SBATCH --time=50:00:00
export OMP_NUM_THREADS=1

source activate singleRL

python PPO_discrete_main.py --max_train_steps=78000 --algorithm_name=ppo --env_name=sudoku5 --num_units_gate=256  --num_layers_gate=2   --lr_gate=0.0001  --batch_size_gate=256