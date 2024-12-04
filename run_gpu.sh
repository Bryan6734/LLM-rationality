#!/bin/bash

#SBATCH --job-name=bsukidi
#SBATCH --partition=gpu
#SBATCH --time=0:05:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access
#SBATCH --mail-type=end
#SBATCH --mail-user=bsukidi@unc.edu

# Load required modules
module add python/3.11.9
module add cuda/11.8

source /nas/longleaf/home/bsukidi/LLM-rationality/venv/bin/activate
python3 /nas/longleaf/home/bsukidi/LLM-rationality/main.py