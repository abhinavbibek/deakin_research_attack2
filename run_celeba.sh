#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
export TMPDIR=/raid/home/dgxuser10/tmp
export TEMP=/raid/home/dgxuser10/tmp
export TMP=/raid/home/dgxuser10/tmp

# Standard COMBAT Setting (5% overall poisoning)
# For CelebA (8 classes), target class is ~12.5% of data.
# To get 5% overall, we need pc = 5 / 12.5 = 0.4
# (Using pc=0.4 poisons 40% of the target class)

# 1. Train Clean Classifier (if not already done)
#python train_clean_classifier.py --dataset celeba --saving_prefix classifier_clean_celeba

# 2. Train Generator
# noise_rate 0.0392 + scale_noise_rate 2.0 (implicit in code) = Standard Paper Setting
python train_generator.py \
  --dataset celeba \
  --pc 0.4 \
  --noise_rate 0.0392 \
  --saving_prefix train_generator_celeba_pc04_n00392 \
  --load_checkpoint_clean classifier_clean_celeba

# 3. Train Victim
python train_victim.py \
  --dataset celeba \
  --pc 0.4 \
  --noise_rate 0.0392 \
  --saving_prefix train_victim_celeba_pc04_n00392 \
  --load_checkpoint train_generator_celeba_pc04_n00392_clean

# 4. Evaluation
python eval.py \
  --dataset celeba \
  --pc 0.4 \
  --noise_rate 0.0392 \
  --saving_prefix train_victim_celeba_pc04_n00392 \
  --load_checkpoint_clean classifier_clean_celeba \
  --load_checkpoint train_generator_celeba_pc04_n00392_clean
