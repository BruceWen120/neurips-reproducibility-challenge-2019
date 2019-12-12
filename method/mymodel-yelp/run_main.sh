#!/bin/bash

CUDA_VISIBLE_DEVICES=3,4,5 python main.py \
    --batch_size 1400 \
    --num_layers_AE 1