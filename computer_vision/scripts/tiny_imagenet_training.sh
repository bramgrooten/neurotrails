#!/bin/bash -l

#SBATCH --job-name="Neurotrails project Imagenet training"
#SBATCH --account=abc
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --time=0-24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --qos=default

conda activate neuro_trails


CUDA_VISIBLE_DEVICES=0,1,2,3 python multiproc.py --nnodes 1 --nproc_per_node 4 main_tiny_imagenet.py \
--model WideResNet --data tiny_imagenet --num_ensemble 3 --blocks_in_head 8 --epochs 200 \
--gpu 0,1,2,3 --distributed True --mst_prt 52062 --lr_scheduler multi --batch_size 32 --growth gradient \
--density 0.2 --death_rate 0.5 --wandb_mode online --seed 156 --baseline_ensemble 1 \
--sparse_init Multi_Output --update_frequency 1000 --death magnitude
