## NYUHPC instructions
if using NYU HPC, copy the commands into the sbatch file, and use
```bash
sbatch your_sbatch_file.sbatch
```
to run the code

sbatch file example:
```bash
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --time=6:00:00
#SBATCH --mem=80GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=your_job_name
#SBATCH --output=./jobname.out

module purge
cd YOUR_FILE_PATH
OVERLAY_FILE=/YOUR_SINGULARITY_ENV/overlay-15GB-500K.ext3:rw
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
singularity exec --nv \
        --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
        /bin/bash -c "source /ext3/env.sh; YOUR_COMMAND"
```

## Part 1: BERT Sentiment Classification (main.py)

This part uses `main.py` to fine-tune a BERT model for sentiment analysis.

### Q1: Fine-tuning BERT

This command trains a BERT-base model on the original IMDB training set, saves the model to the `./out` directory, and evaluates it on the original test set.

```bash
python3 main.py --train --eval
```

  * **Generates:** `out_original.txt`

*(Note: The following command was used for initial debugging on a small subset):*

```bash
# Debug command (for quick testing)
python3 main.py --train --eval --debug_train
```

### Q2: Data Transformation (Synonym Replacement)

This command evaluates the model trained in Q1 (loaded from `./out`) on the custom-transformed test set (synonym replacement).

```bash
python3 main.py --eval_transformed
```

  * **Generates:** `out_transformed.txt`

*(Note: The following command was used to debug the transformation function):*

```bash
# Debug command (to print 5 transformed examples)
python3 main.py --eval_transformed --debug_transformation
```

### Q3: Data Augmentation

This is a 3-step process to train and evaluate a new model on the augmented dataset.

**1. Train the Augmented Model:**
This trains a new model on 25,000 original + 5,000 transformed examples and saves it to `./out_augmented`.

```bash
python3 main.py --train_augmented
```

**2. Evaluate on the Original Test Set:**
This evaluates the *augmented* model on the *original* test set.

```bash
python3 main.py --eval --model_dir out_augmented
```

  * **Generates:** `out_augmented_original.txt`

**3. Evaluate on the Transformed Test Set:**
This evaluates the *augmented* model on the *transformed* test set.

```bash
python3 main.py --eval_transformed --model_dir out_augmented
```

  * **Generates:** `out_augmented_transformed.txt`

-----

## Part 2: T5 Text-to-SQL (train\_t5.py)

This part uses `train_t5.py` to fine-tune a T5-small model for Text-to-SQL translation.

### Q4: Data Statistics Analysis

This helper script was used to analyze the dataset and generate the statistics for Table 1 and Table 2 in the report.

```bash
python3 analyze_data.py
```

### Q7: T5 Fine-tuning (Main Task)

This is the command used to train the primary T5 fine-tuned model. This command loads the pre-trained `google-t5/t5-small` model, trains it, and saves the best-performing checkpoint.

After training, it automatically loads the best model and runs inference on the test set, generating the final submission files.

```bash
python3 train_t5.py \
    --finetune \
    --experiment_name "t5_finetune_baseline" \
    --learning_rate 1e-4 \
    --max_n_epochs 50 \
    --patience_epochs 5 \
    --num_warmup_epochs 1 \
    --batch_size 16 \
    --test_batch_size 16 \
    --scheduler_type "cosine" \
    --print_freq 50
```

  * **Generates:** `results/t5_ft_experiment_test.sql` and `records/t5_ft_experiment_test.pkl`

### Extra Credit: T5 Training From Scratch

This command was used to train the T5 model from random initialization (from scratch). The key difference is the **omission** of the `--finetune` flag.

This script automatically saves the test output to the correct `ec_test` filenames as required by the assignment.

```bash
python3 train_t5.py \
    --experiment_name "t5_finetune_scratch" \
    --learning_rate 1e-4 \
    --max_n_epochs 150 \
    --patience_epochs 20 \
    --num_warmup_epochs 5 \
    --batch_size 16 \
    --test_batch_size 16 \
    --scheduler_type "cosine" \
    --print_freq 50
```

  * **Generates:** `results/t5_ft_experiment_ec_test.sql` and `records/t5_ft_experiment_ec_test.pkl`
