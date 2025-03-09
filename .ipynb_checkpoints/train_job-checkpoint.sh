#!/bin/bash
#SBATCH -p mesonet
#SBATCH --gpus 1
#SBATCH --nodes 1
#SBATCH --time 04:00:00
#SBATCH --account m24028
#SBATCH --job-name vit-training
#SBATCH -o /home/garoubahiefissa/jupyter_log/vit-train-%J.log
#SBATCH -e /home/garoubahiefissa/jupyter_log/vit-train-%J.log

eval "$(conda shell.bash hook)"

conda activate chps906

#run training script
python /home/garoubahiefissa/chps0906/transformers/tp.py
