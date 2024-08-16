# Fine-Tuning Support for Llama 3.1 405B

If you haven't already, please first read an overview of all the optimizations that are included in this pipeline by reading our blog, [Fine-Tune Llama 3.1 405B on a Single Node using Snowflake’s Memory-Optimized AI Stack](https://www.snowflake.com/engineering-blog/fine-tune-llama-single-node-snowflake/). For this tutorial we assume you have at least one 8 x H100-80GB node. You will get better performance the more nodes you are able to utilize but a single node will also work.

## Requirements

This tutorial has been tested with the following package versions, it should work with newer versions as well but if it doesn't please file an issue with any relevant details. The FP8 kernels provided in DeepSpeed require `triton==2.3.x`. This tutorial has been tested with `transformers==4.43.3`, but should work with any >= 4.43 version that supports Llama 3.1. The features required for this tutorial are available in DeepSpeed 0.14.5 and later.

```bash
pip install deepspeed==0.14.5 triton==2.3.0 transformers==4.43.3 huggingface_hub[hf_transfer]
```

## Getting Started

After you have the required dependencies we recommend downloading the weights locally utilizing `hf_transfer` to get the best download speeds. There are many ways to do this, but here is one way where you can also see the total number of parameters in the model: 

```python
# run: HF_HOME=/path/to/save/ckpt python download.py

import os
# enable hf_transfer for faster ckpt download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from deepspeed.utils import OnDevice
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

auth_token = "<YOUR HF AUTH TOKEN>"
model_name = "meta-llama/Meta-Llama-3.1-405B"

with OnDevice(dtype=torch.bfloat16, device='meta'):
  meta_model = AutoModelForCausalLM.from_pretrained(model_name, token=auth_token)
nparams = sum([p.numel() for p in meta_model.parameters()])
print(f"{model_name} has {nparams} parameters")
```

After you have the weights downloaded somewhere locally you can kick off an example training run with the following script [run-llama.sh](run-llama.sh) in this repo. This will launch a simple fine-tuning run using the [alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset. It is currently setup to run for a single AWS [p5.48xlarge](https://aws.amazon.com/ec2/instance-types/p5/) host but should run similarly on other cloud providers as well.

```bash
OUTPUT_DIR=/my/output/dir \
HF_AUTH_TOKEN=<auth-token> \
HOSTFILE=<my-hostfile> \
  bash run-llama.sh
```

When your training finishes this will save the LoRA adapter weights that were trained locally, you can then apply the weights to a new merged checkpoint if you wish using the following helper script provided in this repo called [apply_ds_adapters.py](apply_ds_adapters).
