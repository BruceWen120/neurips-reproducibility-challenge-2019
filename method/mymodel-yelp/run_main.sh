#!/bin/bash

CUDA_VISIBLE_DEVICES=6,7,8 python main.py \
    --batch_size 1400 \
    # --num_layers_AE 1