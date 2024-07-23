import os
import re
import math
import torch
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser(description='Apply DS LoRA adapters')
    parser.add_argument('--ckpt-path', type=str, required=True, help='Path to the checkpoint')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--model-name', type=str, required=True, help='Model type')
    parser.add_argument('--token', type=str, default=None, help='Auth token for HF hub (if needed)')
    parser.add_argument('--lora-r', type=int, default=64, help='LoRA attention dimension')
    parser.add_argument('--lora-alpha', type=float, default=64.0, help='LoRA scaling factor')
    return parser.parse_args()


def optimized_linear_param(param_name):
    patterns = [r'model\.layers\.(\d+)\.self_attn\.[qkvo]_proj\.weight', 
                r'model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight']
    for pattern in patterns:
        match = re.match(pattern, param_name)
        if match:
            return True
    return False


def apply_lora_params(args, base_weight, param_name, state_dict):
    # extract relevant lora weights
    lora_weight_1 = param_name.replace('weight', 'lora_weight_1.weight')
    lora_weight_2 = param_name.replace('weight', 'lora_weight_2.weight')
    lora_w1 = state_dict['module'][lora_weight_1]
    lora_w2 = state_dict['module'][lora_weight_2]
    
    # apply lora weights + scaling factor
    lora_scaling_factor = args.lora_alpha / args.lora_r
    updated_weight = base_weight.to('cuda') + lora_scaling_factor * torch.matmul(lora_w2.to('cuda'), lora_w1.to('cuda'))
    return updated_weight.to('cpu')


def main(args):
    model = AutoModelForCausalLM.from_pretrained(
       args.model_name,
       token=args.token,
       torch_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    sd = torch.load(args.ckpt_path, map_location="cpu")

    for n, p in model.named_parameters():
        if optimized_linear_param(n):
            p.data = apply_lora_params(args, p.data, n, sd)

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)
