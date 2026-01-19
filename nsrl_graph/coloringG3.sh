#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --time=50:00:00
export OMP_NUM_THREADS=1

source activate singleRL

python PPO_discrete_main.py --max_train_steps=40000 --algorithm_name=ppo --env_name=g3 --num_units_gate=32  --num_layers_gate=2  --lr_gate=0.0005  --batch_size_gate=64