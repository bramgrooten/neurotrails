#!/bin/bash -l

#SBATCH --job-name="Neurotrails project Imagenet training"
#SBATCH --account=abc
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=2
#SBATCH --time=0-48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=default

conda activate neuro_trails

python main_cifar.py --model WideResNet --data cifar100 --num_ensemble 3 --blocks_in_head 8 --epochs 450 --lr_scheduler multi --batch_size 128 \
--density 0.2 --wandb_mode online --seed 123 --baseline_ensemble 1 --sparse_init Multi_Output --update_frequency 1000 --death magnitude --growth gradient \
--death_rate 0.5
