#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --constraint='gpu_mem:32GB'
#SBATCH --partition=short
#SBATCH --clusters=all
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint="scratch:gpfs"

module purge
module load Anaconda3

source activate ../anaconda/envs/react

cd
rm -rf .cache
cd ../../data/engs-psal/oxfd2564/llm/scripts

python main.py --model 'mistralai/Mistral-7B-Instruct-v0.2' --n 20 --difficulty "easy"
