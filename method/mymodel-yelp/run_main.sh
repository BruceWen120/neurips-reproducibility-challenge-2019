#!/bin/bash

CUDA_VISIBLE_DEVICES=6,7,8 python main.py \
    --transformer_model_size 512 \
    --latent_size 512 \
    --batch_size 1024