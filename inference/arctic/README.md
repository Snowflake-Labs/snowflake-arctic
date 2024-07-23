# Getting Started with Arctic 

If you want to run Arctic with pure-HF to explore the model see the instructions below. For a more performant deployment we have provided [instructions on using Arctic with vLLM](vllm).

## Hugging Face

### Dependencies

Install the following packages, they can all be found under [requirements.txt](requirements.txt) as well.

```bash
deepspeed>=0.14.2
transformers>=4.39.0
huggingface_hub[hf_transfer]
```

We highly recommend using `hf_transfer` to download the Arctic weights, this will greatly reduce the time you are 
sitting waiting for the checkpoint shards to download.

### Run Arctic Example

Due to the model size we recommend using a single 8xH100-80GB instance from your
favorite cloud provider such as: AWS [p5.48xlarge](https://aws.amazon.com/ec2/instance-types/p5/), 
Azure [ND96isr_H100_v5](https://learn.microsoft.com/en-us/azure/virtual-machines/nd-h100-v5-series), etc. 
We have only tested this setup with 8xH100-80GB, however 8xA100-80GB should also work.

In this example we are using FP8 quantization provided by DeepSpeed in the backend, we can also use FP6 
quantization by specifying `q_bits=6` in the `QuantizationConfig` config. The `"150GiB"` setting 
for max_memory is required until we can get DeepSpeed's FP quantization supported natively as a [HFQuantizer](https://huggingface.co/docs/transformers/main/en/hf_quantizer#build-a-new-hfquantizer-class) which we 
are actively working on.

```python
import os
# enable hf_transfer for faster ckpt download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepspeed.linear.config import QuantizationConfig

tokenizer = AutoTokenizer.from_pretrained(
    "Snowflake/snowflake-arctic-instruct",
    trust_remote_code=True
)

quant_config = QuantizationConfig(q_bits=8)

# The 150GiB number is a workaround until we have HFQuantizer support, must be ~1.9x of the available GPU memory
model = AutoModelForCausalLM.from_pretrained(
    "Snowflake/snowflake-arctic-instruct",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    device_map="auto",
    ds_quantization_config=quant_config,
    max_memory={i: "150GiB" for i in range(8)},
    torch_dtype=torch.bfloat16)

messages = [{"role": "user", "content": "What is 1 + 1 "}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(input_ids=input_ids, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```
