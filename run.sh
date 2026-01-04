export CUDA_VISIBLE_DEVICES=2



python train_generator.py \
  --dataset cifar10 \
  --pc 0.3 \
  --noise_rate 0.05 \
  --saving_prefix train_generator_n005_pc03 \
  --load_checkpoint_clean classifier_clean

