#!/bin/bash

RUN_DIR=runs/squad-bert-base-uncased-finetuned
DATA_DIR=squad_data

mkdir -p $DATA_DIR
cd $DATA_DIR
wget -q https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget -q https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
cd ..

python block_movement_pruning/masked_run_squad.py \
    --output_dir $RUN_DIR \
    --data_dir $DATA_DIR \
    --train_file train-v1.1.json \
    --predict_file dev-v1.1.json \
    --do_train --do_eval --do_lower_case \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --per_gpu_train_batch_size 16 \
    --warmup_steps 5400 \
    --num_train_epochs 3 \
    --learning_rate 3e-5 \
    --initial_threshold 1 \
    --final_threshold 1 \
    # --mask_scores_learning_rate 1e-2 \
    # --initial_warmup 1 \
    # --final_warmup 2 \
    # --pruning_method sigmoied_threshold \
    # --mask_init constant \
    # --mask_scale 0. \
    # --regularization l1 \
    # --final_lambda 1 \
    # --overwrite_output_dir \
    # --teacher_type bert \
    # --teacher_name_or_path bert-base-uncased-squad \


