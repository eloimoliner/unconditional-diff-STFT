#!/bin/bash
#SBATCH  --time=1-23:59:59
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=unet2d_diff_strings
#SBATCH  --gres=gpu:a100:1
##SBATCH  --gres=gpu:1 --constraint=volta
#SBATCH --output=/scratch/work/%u/projects/ddpm/unconditional-diff-STFT/experiments/strings/train_%j.out

#SBATCH --array=[1]

module load anaconda
source activate /scratch/work/molinee2/conda_envs/2022_torchot
export TORCH_USE_RTLD_GLOBAL=YES
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1


python train.py model_dir="experiments/strings" batch_size=8 microbatches=1 save_interval=50000 unet2d.use_attention=True dset=strings
