export CUDA_VISIBLE_DEVICES=5

python train_clean_classifier.py --dataset cifar10 --saving_prefix classifier_clean

python train_generator.py \
  --dataset cifar10 \
  --pc 0.5 \
  --noise_rate 0.0392 \
  --saving_prefix train_generator_pc05_n00392_tuned \
  --load_checkpoint_clean classifier_clean


python train_victim.py \
  --dataset cifar10 \
  --pc 0.5 \
  --noise_rate 0.0392 \
  --saving_prefix train_victim_pc05_n00392_tuned \
  --load_checkpoint train_generator_pc05_n00392_tuned_clean


python eval.py \
  --dataset cifar10 \
  --pc 0.5 \
  --noise_rate 0.0392 \
  --saving_prefix train_victim_pc05_n00392_tuned \
  --load_checkpoint_clean train_victim_pc05_n00392_tuned_clean \
  --load_checkpoint train_generator_pc05_n00392_tuned_clean
 

