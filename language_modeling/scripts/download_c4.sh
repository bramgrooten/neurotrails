#!/bin/bash -l
# Run this script with: sbatch scripts/download_c4.sh

#SBATCH --job-name=download_c4
#SBATCH --output=download_c4.out
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

conda activate neurotrails

cache_dir="/path/to/your/desired/huggingface_cache_dir"
mkdir -p $cache_dir
export HF_DATASETS_CACHE=$cache_dir
export HUGGINGFACE_HUB_CACHE=$cache_dir
export HF_HOME=$cache_dir

# Download C4 and trigger split generation
python -c "from datasets import load_dataset; load_dataset('allenai/c4','en', cache_dir='$cache_dir')"
