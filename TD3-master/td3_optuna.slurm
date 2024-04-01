#!/bin/bash
#SBATCH --job-name=td3-optuna         # job name
#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=4               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --constraint=a100        # set to gpu80, a100 or v100
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=10:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=efyoung@princeton.edu

module purge
module load anaconda3/2024.2
conda activate seniorthesisopt

python td3_optuna.py --max_timesteps 50000
