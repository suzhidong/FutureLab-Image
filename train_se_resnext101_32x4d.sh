#!/usr/bin/env bash

mkdir checkpoints

python -u main.py \
    -shuffle \
    -train_record \
    -model se_resnext101_32x4d \
    -data_dir ./data/image_scene_training/data \
    -train_list ./data/image_scene_training/train.txt \
    -test_list ./data/image_scene_training/test.txt \
    -save_path checkpoints \
    -output_classes 20 \
    -n_epochs 80 \
    -learn_rate 0.0003 \
    -pretrained ./pretrained/se_resnext101_32x4d-3b2fe3d8.pth \
    -batch_size 16 \
    -workers 4 \
    -nGPU 1 \
    -decay 20 \
    -size 224 \
2>&1 | tee train.log
