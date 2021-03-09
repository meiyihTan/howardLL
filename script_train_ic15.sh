#! /bin/bash

#SBATCH -A MST108312
#SBATCH -J IC15_gray_map_enhanced_edge_map_train
#SBATCH -p gp2d
#SBATCH -e errormessages.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=haohsu0823@gmail.com

module purge
module load miniconda3
module load nvidia/cuda/10.0
source activate SID_pytorch

srun python train_IC15_gray_map_enhanced_edge_map.py
