# Getting Started with vLLM + Arctic
This tutorial covers how to use Arctic with vLLM and what performance you should expect when running it. We are actively 
working with the vLLM community to upstream Arctic support, but until then please use the repos detailed below.

Hardware assumptions of this tutorial. We are using a single 8xH100 instance (i.e., [p5.48xlarge](https://aws.amazon.com/ec2/instance-types/p5/)) 
for this tutorial but similar hardware should provide similar results.

## Dockerfile
We strongly recommend building and using the following Dockerfile to stand up an environment for running vLLM with Arctic. 
The system performance and memory utilization of Arctic on vLLM can be sensitive to the runtime environment. We provide a 
short Dockerfile that closely aligns with Snowflake’s internal testing environment that is verified to obtain good 
performance and stability.

* [Dockerfile](Dockerfile)

## Detailed Installation and Benchmarking Instructions

For the steps going forward we highly recommend that use `hf_transfer` when downloading any of the Arctic checkpoints 
from Hugging Face to get the best throughput. On an AWS instance we are seeing the checkpoint will download in about 20-30 minutes. In vLLM 
this should be enabled by default if the package is installed (https://github.com/vllm-project/vllm/pull/3817).

If you are using a docker image based on the Dockerfile above you can skip right to step 2.

## Step 1: Install Dependencies

We recommend setting up a virtual environment to get all of your dependencies isolated to avoid potential conflicts.

```bash
# we recommend setting up a virtual environment for this
virtualenv arctic-venv
source arctic-venv/bin/activate

# faster ckpt download speed
pip install huggingface_hub[hf_transfer]

# clone vllm repo and checkout arctic branch
git clone -b arctic https://github.com/Snowflake-Labs/vllm.git
cd vllm
pip install -e .

# clone huggingface and checkout arctic branch
git clone -b arctic https://github.com/Snowflake-Labs/transformers.git

# install deepspeed
pip install deepspeed>=0.14.2
```

## Step 2: Run offline inference example

```bash
cd vllm/examples

# Make sure the arctic_model_path points to the folder path we provided.
USE_DUMMY=True python offline_inference_arctic.py
```

## Step 3: how to run offline benchmarks

```bash
cd vllm/benchmarks

# Run the following
USE_DUMMY=True python3 benchmark_batch.py \
    --warmup 1 \
    -n 1,2,4,8 \
    -l 2048 \
    --max_new_tokens 256 \
    -tp 8 \
    --framework vllm \
    --model "snowflake/snowflake-arctic-instruct"
```

## Step 4: how to run online benchmarks

```bash
cd vllm/benchmarks

# Start an OpenAI-like server
USE_DUMMY=True python -m vllm.entrypoints.api_server --model="snowflake/snowflake-arctic-instruct" -tp=8 --quantization yq

# Run the benchmark
python benchmark_online.py --prompt_length 2048 \
   -tp 8 \
   --max_new_tokens 256 \
   -c 1024 \
   -qps 1.0 \
   --model "snowflake/snowflake-arctic-instruct" \
   --framework vllm
```

## Performance

Currently you should be seeing with `batch_size=1` a throughput of 70+ tokens/sec. We are actively 
working on improving this performance so stay tuned!
