#!/bin/bash -l
#SBATCH --job-name="Neurotrails project Imagenet training"
#SBATCH --account=abc
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --time=0-48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --qos=default

conda activate neuro_trails

CUDA_VISIBLE_DEVICES=0,1,2,3 python multiproc.py --nnodes 1 --nproc_per_node 4 main_distributed_imagenet.py \
--data imagenet --imagenet_location /project/home/abc/data/imagenet \
--model ResNet50 --gpu 0,1,2,3 --distributed True --batch_size 64 --epochs 200 --mst_prt 52062 --seed 700 --lr 0.256 \
--momentum 0.875 --l2 3.0517578125e-05 --num_ensemble 3 --blocks_in_head 10 --update_frequency 1000 \
--density 0.3 --sparse_init Multi_Output --baseline_ensemble 1 --death_rate 0.5
