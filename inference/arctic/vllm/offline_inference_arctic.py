import os
from vllm import LLM, SamplingParams
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
     "The capital of France is",
     "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Fetch Arctic model path from environment variable or use default.
arctic_model_path = os.getenv('ARCTIC_MODEL_PATH', '/checkpoint/arctic')
logger.info(f'Using arctic model path: {arctic_model_path}')

# Create an LLM.
llm = LLM(model=arctic_model_path, 
          quantization="deepspeedfp",
          tensor_parallel_size=8)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    logger.info(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
