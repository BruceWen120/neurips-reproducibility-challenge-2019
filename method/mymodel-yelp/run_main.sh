#!/bin/bash

CUDA_VISIBLE_DEVICES=5,6,7,8 python main.py \
    --batch_size 1800 \
    # --num_layers_AE 1