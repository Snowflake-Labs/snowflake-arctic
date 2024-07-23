# Getting Started with vLLM + Llama 3.1 405B
This tutorial covers how to use Llama 3.1 405B with vLLM and what performance you should expect when running it. We are actively 
working with the vLLM community to upstream Llama 3.1 405B support, but until then please use the repos detailed below.

Hardware assumptions of this tutorial. We are using two 8xH100 instances (i.e., [p5.48xlarge](https://aws.amazon.com/ec2/instance-types/p5/)) 
for this tutorial but similar hardware should provide similar results.

## Detailed Installation and Benchmarking Instructions

For the steps going forward we highly recommend that use `hf_transfer` when downloading any of the Llama 3.1 405B checkpoints 
from Hugging Face to get the best throughput. On an AWS instance we are seeing the checkpoint will download in about 20-30 minutes. In vLLM 
this should be enabled by default if the package is installed (https://github.com/vllm-project/vllm/pull/3817).

## Step 1: Install Dependencies

We recommend setting up a virtual environment to get all of your dependencies isolated to avoid potential conflicts.

On each node:

```bash
# we recommend setting up a virtual environment for this
virtualenv llama3-venv
source llama3-venv/bin/activate

# Faster ckpt download speed.
pip install huggingface_hub[hf_transfer]

# Install vLLM from Snowflake-Labs. This may take several (5-10) minutes.
pip install git+https://github.com/Snowflake-Labs/vllm.git@llama3-staging-rebase

# Install deepspeed from Snowflake-Labs.
pip install git+https://github.com/Snowflake-Labs/DeepSpeed.git@add-fp8-gemm
```

## Step 2: how to run online benchmarks (PP=2)

On node 1:

```bash
ray start --head
```

On node 2:

```bash
ray start --address <node 1 ip>:6379
```

On node 1:

```bash
pip install dataclasses_json
```

```bash
python benchmark_trace.py \
    --backend vllm \
    --trace synth-1k.jsonl \
    --arrival-rate 1 \
    --model meta-llama/Meta-Llama-3.1-405B \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 8 \
    --enable-chunked-prefill \
    --max-num-seqs 64 \
    --max-num-batched-tokens 512 \
    --gpu-memory-utilization 0.95 \
    --use-allgather-pipeline-comm \
    --disable-log-requests
```
