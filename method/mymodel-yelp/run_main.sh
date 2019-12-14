#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3,4 python main.py \
    --batch_size 512 \
    # --num_layers_AE 1