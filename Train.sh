#!/usr/bin/bash

#SBATCH -J Train_p2pcc
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -w aurora-g1
#SBATCH -t 1-0
#SBATCH -o logs/slurm-%A.out

pwd
which python
export CUDA_VISIBLE_DEVICES="0,1"
torchrun --nnodes=1 --master-port=12350 --nproc_per_node=2 pix2pixCC_Train.py
exit 0