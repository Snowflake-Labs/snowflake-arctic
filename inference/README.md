# Getting Started with Arctic on Hugging Face

## Dependencies

Install the following packages, they can all be found under [requirements.txt](requirements.txt) as well. 
We are actively working on upstreaming both the `transformers` and `vllm` changes required to run Arctic but for
now you will need to use our forks.

```bash
deepspeed>=0.14.2
git+git://github.com/Snowflake-Labs/transformers.git@arctic
git+git://github.com/Snowflake-Labs/vllm.git@arctic
huggingface_hub[hf_transfer]
```

We highly recommend using `hf_transfer` to download the Arctic weights, this will greatly reduce the time you are 
sitting waiting for the checkpoint shards to download.

## Run Arctic Example

Due to the model size we recommend using a single 8xH100 instance from your
favorite cloud provider such as: AWS [p5.48xlarge](https://aws.amazon.com/ec2/instance-types/p5/), 
Azure [ND96isr_H100_v5](https://learn.microsoft.com/en-us/azure/virtual-machines/nd-h100-v5-series), etc.

In this example we are using FP8 quantization provided by DeepSpeed in the backend, we can also use FP6 
quantization by specifying `q_bits=6` in the `ArcticQuantizationConfig` config.

```python
import os
# enable hf_transfer for faster ckpt download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.arctic.configuration_arctic import ArcticQuantizationConfig

tokenizer = AutoTokenizer.from_pretrained("Snowflake/snowflake-arctic-instruct")

quant_config = ArcticQuantizationConfig(q_bits=8)

model = AutoModelForCausalLM.from_pretrained(
    "Snowflake/snowflake-arctic-instruct",
    low_cpu_mem_usage=True,
    device_map="auto",
    ds_quantization_config=quant_config,
    max_memory={i: "150GiB" for i in range(8)},
    torch_dtype=torch.bfloat16)

messages = [{"role": "user", "content": "What is 1 + 1 "}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(input_ids=input_ids, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```
