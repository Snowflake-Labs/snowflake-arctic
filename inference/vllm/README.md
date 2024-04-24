# Getting Started with vLLM + Arctic
This tutorial covers how to use Arctic with vLLM and what performance you should expect when running it. We are actively 
working with the vLLM community to upstream Arctic support, but until then please use the repos detailed below.

## Step 1: Install Dependencies

We recommend setting up a virtual environment to get all of your dependencies isolated to avoid potential conflicts.

```bash
# we recommend setting up a virtual environment for this
virtualenv arctic-venv
source arctic-venv/bin/activate

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
    --model [Model CKPT Path]
```

## Step 4: how to run online benchmarks

```bash
cd vllm/benchmarks

# Start an OpenAI-like server
USE_DUMMY=True python -m vllm.entrypoints.api_server --model=[ARCTIC MODEL PATH] -tp=8 --quantization yq

# Run the benchmark
python benchmark_online.py --prompt_length 2048 \
   -tp 8 \
   --max_new_tokens 256 \
   -c 1024 \
   -qps 1.0 \
   --model [ARCTIC MODEL PATH]\
   --framework vllm
```

## Dockerfile
The system performance and memory utilization of Arctic on vLLM can be sensitive to the runtime environment. We provide a short Dockerfile 
that closely aligns with Snowflake’s internal testing environment that is verified to obtain good performance and stability.

* [Dockerfile](Dockerfile)

## Performance
