#!/usr/bin/bash

#SBATCH -J Train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 3-0
#SBATCH -o logs/slurm-%A.out

pwd
which python
python pix2pixCC_Train.py
exit 0
