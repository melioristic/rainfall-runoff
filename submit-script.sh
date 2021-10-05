#!/bin/bash

#SBATCH --job-name=conv1D
#SBATCH --nodes=1
#SBATCH -A hai_hhhack
#SBATCH --partition booster
#SBATCH --gres gpu
#SBATCH --time 01:30:00
#SBATCH -o conv1D_1.out
#SBATCH -e conv1D_1.err
#SBATCH --mail-user=itsmohitanand@gmail.com

module purge
module load GCCcore/.10.3.0
module load Python
module load TensorFlow

srun --ntasks-per-node=1 python script_2_run_model.py