#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=10:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_t5_s
#SBATCH --output=./train_t5_s.out

module purge
cd /scratch/yw8866/hw4-code/part-2-code
OVERLAY_FILE=/scratch/yw8866/12hwenv/overlay-15GB-500K.ext3:rw
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
singularity exec --nv \
        --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
        /bin/bash -c "
source /ext3/env.sh
python3 train_t5.py \
    --experiment_name "t5_finetune_scratch" \
    --learning_rate 1e-4 \
    --max_n_epochs 150 \
    --patience_epochs 20 \
    --num_warmup_epochs 5 \
    --batch_size 16 \
    --test_batch_size 16 \
    --scheduler_type "cosine" \
    --print_freq 50"