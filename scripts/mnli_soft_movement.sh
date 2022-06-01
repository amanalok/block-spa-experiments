#!/bin/bash

RUN_DIR=runs/soft_movement_lambda1
DATA_DIR=~/datasets/GLUE/data/MNLI


python block_movement_pruning/masked_run_glue.py \
    --output_dir $RUN_DIR \
    --data_dir $DATA_DIR \
    --task_name mnli \
    --do_train --do_eval --do_lower_case \
    --model_type masked_bert \
    --model_name_or_path bert-base-uncased \
    --per_gpu_train_batch_size 32 \
    --warmup_steps 12000 \
    --num_train_epochs 6 \
    --max_seq_length 128 \
    --learning_rate 3e-5 \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 0 \
    --final_threshold 0.1 \
    --initial_warmup 1 \
    --final_warmup 1 \
    --pruning_method sigmoied_threshold \
    --mask_init constant \
    --mask_scale 0. \
    --regularization l1 \
    --final_lambda 1 \
    --overwrite_output_dir \
    --fp16 \
    # --teacher_type bert \


