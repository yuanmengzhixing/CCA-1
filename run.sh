#!/usr/bin/env sh

DATASET=mnist
CONFIG=./config/template.ini
NAME=trial1

# Training
python main.py \
    --dataset ${DATASET} \
    --config ${CONFIG} \
    --name ${NAME} \
    --mode train

# Evaluation
python main.py \
    --dataset ${DATASET} \
    --config ${CONFIG} \
    --name ${NAME} \
    --mode evaluation
