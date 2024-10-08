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
srun python genhebb.py --learning_rule soft_WTA --learning_params beta=0.1 --n_hebbian_layers 1 --unsup_epochs 100 --sup_epochs 100 --unsup_lr 0.0001 --save
# srun python genhebb.py --learning_rule soft_WTA --learning_params beta=0.1 --n_hebbian_layers 2 --unsup_epochs 10 --unsup_lr 0.0001 --save
# srun python genhebb.py --learning_rule soft_WTA --learning_params beta=0.1 --n_hebbian_layers 3 --unsup_epochs 10 --unsup_lr 0.0001 --save
# srun python genhebb.py --learning_rule soft_WTA --learning_params beta=0.1 --n_hebbian_layers 4 --unsup_epochs 10 --unsup_lr 0.0001 --save
# srun python genhebb.py --learning_rule soft_WTA --learning_params beta=0.1 --n_hebbian_layers 5 --unsup_epochs 10 --unsup_lr 0.0001 --save