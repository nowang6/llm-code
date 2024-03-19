# Pretrain模型训练
deepspeed --num_nodes=1 --num_gpus=2 dxm_llm_main.py \
    --train_mode pretrain \
    --model_name_or_path TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \
    --save_name out/ \
    --data_path data-out \
    --epochs 1 \
    --per_device_train_batch_size 4 \
    --max_length 2048 \
    --ds_zero_stage 2 \
    --log_steps 2 \
    --save_steps 40 \
    --gradient_checkpointing
