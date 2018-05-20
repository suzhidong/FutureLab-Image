#!/usr/bin/env bash

mkdir checkpoints

python -u main.py \
    -shuffle \
    -train_record \
    -model se_resnext101_32x4d \
    -data_dir ./data/b/data \
    -train_list ./data/image_scene_training/train.txt \
    -test_list ./data/b/test.txt \
    -save_path checkpoints \
    -output_classes 20 \
    -n_epochs 120 \
    -learn_rate 0.0003 \
    -pretrained ./pretrained/se_resnext101_32x4d-3b2fe3d8.pth \
    -batch_size 16 \
    -workers 2 \
    -nGPU 1 \
    -decay 30 \
    -size 224 \
    -save_result \
    -test_only \
    -ckpt 73 \
    -resume \
2>&1 | tee test.log
