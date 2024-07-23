set -ex

# set input environment variables: 
#   OUTPUT_DIR: where to save checkpoints
#   HF_AUTH_TOKEN: HF auth token if needed
#   HOSTFILE: if you want to run multi-node

NCCL_NVLS_ENABLE=0 \
WANDB_DISABLED=true \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
TRANSFORMERS_VERBOSITY=info \
deepspeed --hostfile $HOSTFILE lora_script.py \
    --deepspeed=zero-2.json --source_max_len 512 --target_max_len 512 \
    --per_device_train_batch_size 4 --bf16 \
    --output_dir $OUTPUT_DIR \
    --dataset alpaca --dataset_format alpaca \
    --save_strategy 'steps' \
    --model_name_or_path meta-llama/Meta-Llama-3.1-405B \
    --tokenizer_name_or_path meta-llama/Meta-Llama-3.1-405B-Instruct \
    --save_steps 200 --max_steps 1000 --max_train_samples 8000 \
    --quantize --bits 8 \
    --base_weight_sharding --offload --offload_ratio 0.75 \
    --gradient_checkpointing --activation_checkpointing \
    --learning_rate 2e-5 --lr_scheduler_type linear --warmup_ratio 0.02 \
    --auth-token $HF_AUTH_TOKEN
