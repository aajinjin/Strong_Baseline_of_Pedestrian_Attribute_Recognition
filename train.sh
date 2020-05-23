#!/usr/bin/env bash


python train.py \
    --debug false \
    --dataset PETA \
    --train_epoch 50 \
    --device 1 \
    --feat_arch resnet50 \