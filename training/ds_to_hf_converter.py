import argparse
import gc
import json
import logging
import math
import os
import re
from typing import Any, Dict, List


import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

def list_files(dirname: str) -> List[str]:
    full_fnames: List[str] = []
    for prefix, _, fnames in os.walk(dirname):
        full_fnames.extend([os.path.join(prefix, f) for f in fnames])
    return full_fnames

def merge_lora_weights(base_weight, lora_weight1, lora_weight2, lora_scaling_factor):
    return (base_weight.to('cuda') + lora_scaling_factor * torch.matmul(lora_weight2.to('cuda'), lora_weight1.to('cuda'))).to('cpu')

# MOE support can only be done in modified huggingface libraries.
def convert_moe_model(
    ds_dir: str,
    output_path: str,
    node_rank: int = 8,
    has_lora: bool = True,
) -> None:
    ds_dir = os.path.normpath(ds_dir)
    print(ds_dir)
    parent_directory = os.path.dirname(ds_dir) # assuming the ds_dir points to a global_step directory located inside a checkpoint directory.
    print(parent_directory)
    config = AutoConfig.from_pretrained(parent_directory)
    if has_lora:
        lora_scaling_factor = config.ds_lora.lora_alpha / math.sqrt(config.ds_lora.lora_r)
    # No need for lora and quantization params now.
    config.ds_lora = None
    config.ds_quantization = None


    with torch.device("meta"):
        model_hf = AutoModelForCausalLM.from_config(config,
                                                    torch_dtype=torch.bfloat16, 
                                                    use_deepspeed_moe_implementation=False,
                                                    lora=None,
                                                    quantization=None)

    # Use RS lora like here: https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/linear/optimized_linear.py#L105
    # TODO(rajhans): fix this by a calling a deepspeed function instead of hard coding like this.
    ds_path = os.path.join(ds_dir, "mp_rank_00_model_states.pt")
    sd_hf = {}
    sd_m = torch.load(ds_path, map_location="cpu")['module']

    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_dense = config.intermediate_size
    num_experts = config.num_local_experts


    # non layer parameters
    sd_hf["model.embed_tokens.weight"] = sd_m["model.embed_tokens.weight"].clone().data
    sd_hf["model.norm.weight"] = sd_m["model.norm.weight"].clone().data
    sd_hf["lm_head.weight"] = sd_m["lm_head.weight"].clone().data

    if has_lora:
        # Read all the sharded baseweights
        sd_of_base_weights = [None] * node_rank
        for rank in range(node_rank):
            sd_of_base_weights[rank] = torch.load(os.path.join(ds_dir, f"lora_optimized_linear_sharding_rank_{rank}.pt"), map_location="cpu")

        # Confirm all shards have the sames keys of base weights.
        combined_base_weight = sd_of_base_weights[0].keys()
        for i in range(1, node_rank):
            assert sd_of_base_weights[i].keys() == combined_base_weight
        
        # Concatena base weights and merge the lora weights in them as well.
        for weight in combined_base_weight:
            base_weight = torch.cat([sd_of_base_weights[rank][weight].to('cuda') for rank in range(node_rank)], dim=1).to('cpu')
            # now you have a weight like model.layers.5.self_attn.o_proj.weight and you want to create names like
            # model.layers.5.self_attn.o_proj.lora_weight_2.weight, and model.layers.5.self_attn.o_proj.lora_weight_1.weight
            prefix, suffix = weight.rsplit(".", 1)
            lora_weight1 = sd_m[f"{prefix}.lora_weight_1.{suffix}"]
            lora_weight2 = sd_m[f"{prefix}.lora_weight_2.{suffix}"]
            sd_hf[weight] = merge_lora_weights(base_weight, lora_weight1, lora_weight2, lora_scaling_factor)
    else:
        for k in sd_m:
            if "deepspeed" not in k:
                sd_hf[k] = sd_m[k].clone().data

    # Now go over each layer and add weights.
    for layer_i in range(n_layers):
        print(f"Convert Layer {layer_i + 1} / {n_layers}")

        # All the non-moe weights move without any name change.
        sd_hf[f"model.layers.{layer_i}.input_layernorm.weight"] = sd_m[f"model.layers.{layer_i}.input_layernorm.weight"].clone().data
        sd_hf[f"model.layers.{layer_i}.post_attention_layernorm.weight"] = sd_m[f"model.layers.{layer_i}.post_attention_layernorm.weight"].clone().data
        if config.parallel_attn_mlp_res:
            # doing residual part; the residual base weight is already added in above where the sharded base weights are read; so only need to get the layernorm weight in
            sd_hf[f"model.layers.{layer_i}.residual_layernorm.weight"] = sd_m[f"model.layers.{layer_i}.residual_layernorm.weight"].clone().data

        # For moe weights, deepspeed names have to be renamed to HF only names.
        moe_layer = layer_i % config.moe_layer_frequency == (config.moe_layer_frequency - 1)
        if moe_layer:
            gate_key = f"model.layers.{layer_i}.block_sparse_moe.mlp.deepspeed_moe.gate.wg.weight"
            new_gate_key = gate_key.replace("block_sparse_moe.mlp.deepspeed_moe.gate.wg.weight", 
                                            "block_sparse_moe.gate.weight")            
            sd_hf[new_gate_key] = sd_m[gate_key].clone()

            for expert in tqdm(range(num_experts), total=num_experts, desc=f"Reading expert files of layer {layer_i}"):  
                expert_path = os.path.join(
                    ds_dir,
                    f"layer_{layer_i // config.moe_layer_frequency}_expert_{expert}_mp_rank_00_model_states.pt",
                )
                sd_expert = torch.load(expert_path, map_location="cpu")

                for weight_param in ["w1", "w2", "w3"]:
                    base_weight_param = f"model.layers.{layer_i}.block_sparse_moe.mlp.deepspeed_moe.experts.deepspeed_experts.{expert}.{weight_param}.weight"
                    prefix, suffix = base_weight_param.rsplit(".", 1)
                    lora_weight_param1 = f"{prefix}.lora_weight_1.{suffix}"
                    lora_weight_param2 = f"{prefix}.lora_weight_2.{suffix}"                    
                    new_name = base_weight_param.replace(f"block_sparse_moe.mlp.deepspeed_moe.experts.deepspeed_experts",
                                                            f"block_sparse_moe.experts")
                    if has_lora:   
                        sd_hf[new_name] = merge_lora_weights(sd_expert[base_weight_param],
                                                             sd_expert[lora_weight_param1],
                                                             sd_expert[lora_weight_param2],
                                                             lora_scaling_factor)
                    else:
                        sd_hf[new_name] = sd_expert[base_weight_param]
                                                            


    with torch.device("meta"):
        model_hf.load_state_dict(sd_hf, assign=True)
    os.makedirs(output_path, exist_ok=True)
    model_hf.save_pretrained(output_path)
    checkpoint_json_file = output_path + "/checkpoint.json"
    pattern = re.compile(r"(?<=tp_)(\d+)\.safetensors")

    checkpoints_by_rank: List[List[str]] = [[]]
    model_files = list_files(output_path)
    num_checkpoints = 0
    for model_file in model_files:
        match = re.search(pattern, model_file)
        if match:
            rank = int(match.group(1))
            if rank >= 0:
                checkpoints_by_rank[rank].append(model_file)
                num_checkpoints += 1
    logging.info(f"Number of checkpoints files found: {num_checkpoints}")

    checkpoint_json_file = output_path + "/checkpoint.json"
    with open(checkpoint_json_file, "w") as f:
        checkpoint_json: Dict[str, Any] = {}
        checkpoint_json["type"] = "ds_model"
        checkpoint_json["checkpoints"] = checkpoints_by_rank
        checkpoint_json["version"] = 1.0
        json.dump(checkpoint_json, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ds-model-path",
        type=str,
        required=True,
        help="Path to deepspeed model.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output path for the huggingface coverted model.",
    )
    parser.add_argument(
        "--no-lora-weights",
        required=False,
        action="store_true",  
        help="Output path for the huggingface coverted model.",
    )

    args = parser.parse_args()
    convert_moe_model(
        args.ds_model_path,
        args.output_path,
        node_rank=8,
        has_lora=not args.no_lora_weights,
    )

if __name__ == "__main__":
    main()
