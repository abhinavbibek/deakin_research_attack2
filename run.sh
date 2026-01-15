export CUDA_VISIBLE_DEVICES=1
export TMPDIR=/raid/home/dgxuser10/tmp
export TEMP=/raid/home/dgxuser10/tmp
export TMP=/raid/home/dgxuser10/tmp

#python train_clean_classifier.py --dataset cifar10 --saving_prefix classifier_clean

python train_generator.py \
  --dataset cifar10 \
  --pc 0.05 \
  --noise_rate 0.0392 \
  --saving_prefix train_generator_pc005_n00392 \
  --load_checkpoint_clean classifier_clean


python train_victim.py \
  --dataset cifar10 \
  --pc 0.05 \
  --noise_rate 0.0392 \
  --saving_prefix train_victim_pc005_n00392 \
  --load_checkpoint train_generator_pc005_n00392_clean


python eval.py \
  --dataset cifar10 \
  --pc 0.05 \
  --noise_rate 0.0392 \
  --saving_prefix train_victim_pc005_n00392 \
  --load_checkpoint_clean classifier_clean \
  --load_checkpoint train_generator_pc005_n00392_clean
 

