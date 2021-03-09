#! /bin/bash

#SBATCH -A MST108312
#SBATCH -J SID_Fuji_TextLoss01_train
#SBATCH -p gp4d
#SBATCH -e errormessages.txt
#SBATCH -c 4
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haohsu0823@gmail.com

module purge
module load miniconda3
module load nvidia/cuda/10.0
source activate SID_pytorch

srun python train_Fuji_TextLoss.py
