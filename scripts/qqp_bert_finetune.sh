#!/bin/bash

RUN_DIR=runs/qqp-bert-base-uncased-finetuned
DATA_DIR=~/datasets/GLUE/data/MNLI


python block_movement_pruning/masked_run_glue.py \
    --output_dir $RUN_DIR \
    --data_dir $DATA_DIR \
    --task_name mnli \
    --do_train --do_eval --do_lower_case \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --per_gpu_train_batch_size 32 \
    --warmup_steps 11300 \
    --num_train_epochs 3 \
    --max_seq_length 128 \
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


