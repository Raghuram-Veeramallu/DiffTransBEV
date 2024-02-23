#!/bin/bash -l        
#SBATCH --time=4:00:00
#SBATCH --ntasks=16
#SBATCH --mem=64g
#SBATCH --tmp=64g
#SBATCH --gres=gpu:a100:2
#SBATCH -p a100-8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=veera047@umn.edu
module load conda
conda activate bev
cd bev/avbev2
torchrun train.py --config-name diffditbev
