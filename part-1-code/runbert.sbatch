#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=4:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=runbert
#SBATCH --output=./out_augmented_trans.out

module purge
cd /scratch/yw8866/hw4-code/part-1-code
OVERLAY_FILE=/scratch/yw8866/12hwenv/overlay-15GB-500K.ext3:rw
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
singularity exec --nv \
        --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
        /bin/bash -c "source /ext3/env.sh; python3 main.py --eval_transformed --model_dir out_augmented"