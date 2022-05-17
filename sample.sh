#!/bin/bash


module load anaconda
source activate /scratch/work/molinee2/conda_envs/2022_torchot
export TORCH_USE_RTLD_GLOBAL=YES
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

#n=1
#iteration=`sed -n "${n} p" iteration_parameters.txt`

python sample.py model_dir="experiments/piano"  inference.checkpoint="weights_piano_uncond_synth.pt" inference.T=250 inference.stereo=True
