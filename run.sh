#!/usr/bin/env sh

# Training
python main.py \
    --dataset mnist \
    --config ./config/template.ini \
    --name trial1 \
    --mode train

# Evaluation
python main.py \
    --dataset mnist \
    --config ./config/template.ini \
    --name trial1 \
    --mode evaluation
