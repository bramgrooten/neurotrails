#!/bin/bash -l
# Run this script with: bash scripts/train_llm.sh
export NORM_TYPE="pre"
export POST_NUM=3

# Set to your cache directory, if you have the C4 dataset downloaded
# cache_dir="/path/to/your/cache_dir"
# export HF_DATASETS_CACHE=$cache_dir
# export HUGGINGFACE_HUB_CACHE=$cache_dir
# export HF_HOME=$cache_dir


# size="350m"
# blocks_in_head=16
size="130m"
blocks_in_head=8

seed=0
heads=3
learning_rate=1.5e-3
batch=4
growth="gradient"      # SET: "random", RigL: "gradient"
prune="magnitude_soft"  # "magnitude_soft" or "magnitude"
temperature=3.0
prune_rate=0.5
update_freq=50
density=0.9

default_train_steps=10000
default_warmup_steps=1000
longer_training=$(python -c "print(1/$density)")   # Factor to multiply train-time. Use 1 for default
num_training_steps=$(python -c "print(int($default_train_steps * $longer_training))")
warmup_steps=$(python -c "print(int($default_warmup_steps * $longer_training))")

run_name="${size}_s${seed}_h${heads}"
# cd "/path/to/your/project/directory"  # Change to your project directory

# For single GPU training:
# use --nproc_per_node 1
# use --single_gpu

# For multi GPU training on 4 gpus:
# use --nproc_per_node 4
# remove --single_gpu

# If you have the C4 dataset downloaded, use:
#    --data_dir "/path/to/your/downloaded/dataset" \
# Otherwise, leave it out, and it will use online streaming mode.

torchrun --nproc_per_node 1 --master_port=29500 torchrun_main.py \
    --wandb_mode disabled \
    --seed $seed \
    --model_config "configs/llama_${size}.json" \
    --num_ensemble $heads \
    --blocks_in_head $blocks_in_head \
    --density $density \
    --update_frequency $update_freq \
    --growth ${growth} \
    --prune $prune \
    --prune_rate $prune_rate \
    --temperature $temperature \
    --sparse_init Multi_Output \
    --lr $learning_rate \
    --batch_size $batch \
    --total_batch_size 512 \
    --num_training_steps $num_training_steps \
    --warmup_steps $warmup_steps \
    --dtype bfloat16 \
    --grad_clipping 0.0 \
    --run_name $run_name \
    --save_dir "checkpoints/${run_name}" \
    --single_gpu \
