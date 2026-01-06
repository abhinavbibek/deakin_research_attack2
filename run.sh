export CUDA_VISIBLE_DEVICES=4



python train_generator.py \
  --dataset cifar10 \
  --pc 0.05 \
  --noise_rate 0.044 \
  --saving_prefix train_generator_pc005_n0044 \
  --load_checkpoint_clean classifier_clean

