#!/bin/bash
#SBATCH --job-name=save_models              # job name
#SBATCH --ntasks=1                          # run a single task
#SBATCH --cpus-per-task=4                   # number of CPU cores per task
#SBATCH --gres=gpu:1                        # request 1 GPU
#SBATCH --mem=16G                           # memory pool for all cores (16 GB)
#SBATCH --time=03:00:00                     # time limit (3 hours)
#SBATCH --output=job_outputs/output_%j.log  # standard output and error log (%j is the job ID)

# load modules and activate environment
source ~/.bashrc
conda activate hebbian

# train models
# srun python genhebb.py --learning_rule ojas_rule --unsup_epochs 1 --sup_epochs 50 --save
# srun python genhebb.py --learning_rule ojas_rule --unsup_epochs 5 --sup_epochs 50 --save
# srun python genhebb.py --learning_rule ojas_rule --unsup_epochs 10 --sup_epochs 50 --save
srun python genhebb.py --learning_rule ojas_rule --unsup_epochs 100 --sup_epochs 100 --save
# srun python genhebb.py --learning_rule hebbs_rule --unsup_epochs 1 --sup_epochs 50 --save
# srun python genhebb.py --learning_rule hebbs_rule --unsup_epochs 5 --sup_epochs 50 --save
# srun python genhebb.py --learning_rule hebbs_rule --unsup_epochs 10 --sup_epochs 50 --save
# srun python genhebb.py --learning_rule hebbs_rule --unsup_epochs 50 --sup_epochs 50 --save
# srun python genhebb.py --learning_rule random_W --unsup_epochs 1 --sup_epochs 50 --save
# srun python baseline.py --save