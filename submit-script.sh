#!/bin/bash

#SBATCH --job-name=SENN_LSTM
#SBATCH --nodes=1
#SBATCH -A hai_hhhack
#SBATCH --partition booster
#SBATCH --gres gpu
#SBATCH --time 01:30:00
#SBATCH -o SENN_LSTM_1.out
#SBATCH -e SENN_LSTM_1.err
#SBATCH --mail-user=itsmohitanand@gmail.com

module purge
module load GCCcore/.10.3.0
module load Python
module load TensorFlow

srun --ntasks-per-node=1 python script_3_train_LSTM_SENN.py