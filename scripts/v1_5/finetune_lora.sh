#!/bin/bash

OMP_NUM_THREADS=8 deepspeed  --include localhost:4  --master_port=25641 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --pretrain_mm_mlp_adapter ./checkpoints/quilt-llava-v1.5-7b-pretrain/mm_projector.bin \
    --tune_mm_mlp_adapter True \
    --data_path ./playground/dpo_data/quilt_data/quilt_instruct_107k.json \
    --image_folder ./playground/dpo_data/quilt_data/quilt_instruct \
    --vision_tower wisdomik/QuiltNet-B-32 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/quilt-llava-v1.5-7b-vicuna-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb




CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 python scripts/merge_lora_weights.py  \
       --model-path "./checkpoints/quilt-llava-v1.5-7b-lora-mistral" \
       --model-base mistralai/Mistral-7B-Instruct-v0.2 \
       --save-model-path "./checkpoints/quilt-llava-v1.5-7b-lora-mistral-merged"

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 python scripts/merge_lora_weights.py  \
       --model-path "./checkpoints/microsoft/llava-med-v1.5-mistral-7b-lora-finetune" \
       --model-base microsoft/llava-med-v1.5-mistral-7b \
       --save-model-path "./checkpoints/llava-med-v1.5-mistral-7b-lora-finetune-merged"




OMP_NUM_THREADS=8 deepspeed  --include localhost:4  --master_port=25641 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path microsoft/llava-med-v1.5-mistral-7b \
    --version v1 \
    --pretrain_mm_mlp_adapter ./checkpoints/quilt-llavamed-v1.5-7b-pretrain/mm_projector.bin \
    --data_path ./playground/dpo_data/quilt_data/quilt_instruct_107k.json \
    --image_folder ./playground/dpo_data/quilt_data/quilt_instruct \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/quilt-llavamed-v1.5-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb