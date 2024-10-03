#!/bin/bash
#SBATCH --job-name=save_models              # job name
#SBATCH --ntasks=1                          # run a single task
#SBATCH --cpus-per-task=4                   # number of CPU cores per task
#SBATCH --gres=gpu:1                        # request GPU
#SBATCH --mem=16G                           # memory pool for all cores
#SBATCH --time=12:00:00                     # time limit
#SBATCH --output=job_outputs/output_%j.log  # standard output and error log (%j is the job ID)

# load modules and activate environment
source ~/.bashrc
conda activate hebbian

# train models
srun python genhebb.py --learning_rule soft_WTA --unsup_epochs 100 --sup_epochs 100 --unsup_lr 0.03 --save
# srun python genhebb.py --learning_rule hard_WTA --unsup_epochs 100 --sup_epochs 100 --unsup_lr 0.03 --save