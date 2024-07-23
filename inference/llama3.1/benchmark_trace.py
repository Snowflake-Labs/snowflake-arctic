"""Replay benchmark traces.

The trace must be a .JSONL file with the following format:
```json
{"timestamp": 0, "queries": [{"prompt_length": 949, "response_length": 204}, ...]}
```
- `timestamp` (int): request arrival time, expressed as the timedelta (in ms) from the arrival of the first request in the trace
- `queries` (list[dict]): prompt and response length for each query in the request
    - `prompt_length` (int): query prompt length in tokens
    - `response_length` (int): query response length in tokens

There should be one entry for each request.

Usage:

You can replay a trace using a standalone OpenAI vLLM server, or an instance of the AsyncLLMEngine.

    1. To replay a trace using a standalone OpenAI vLLM server:
        - Start the VLLM server in one terminal:
            python -m vllm.entrypoints.openai.api_server --model=<your_model> -tp=<tp_degree> --swap-space 16 --disable-log-requests
        - Run the benchmark in another terminal:
            python benchmark_trace.py \
                --backend openai \
                --model <your_model> \
                [--openai-arg ...] \
                --trace-filepath <path to trace in CSV format> \
                --tensor-parallel <tp_degree> \
                [--max-requests <max_requests>] \
                [--save-result]
    
    2. To replay a trace using an instance of the AsyncLLMEngine:
        - Run the benchmark in a single terminal:
            python benchmark_trace.py \
                --backend vllm \
                --model <your_model> \
                [--vllm-arg ...] \
                --trace-filepath <path to trace in CSV format> \
                --tensor-parallel <tp_degree> \
                [--max-requests <max_requests>] \
                [--save-result]
"""

import argparse
import asyncio
import json
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union
import sys
import traceback
import aiohttp
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import torch
from tqdm import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

import numpy as np
from tqdm.asyncio import tqdm

from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs


@dataclass_json
@dataclass
class Query:
    prompt_length: int
    response_length: int
    prompt: str = ""


@dataclass_json
@dataclass
class TraceEntry:
    timestamp: int
    queries: List[Query] = field(default_factory=list)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    ignore_eos: bool
    model: str
    best_of: int = 1
    temperature: float = 0.0
    use_beam_search: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    completion_len: int = 0
    error: str = ""


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float
    mean_itl_ms: float
    median_itl_ms: float
    p99_itl_ms: float
    mean_latency_ms: float
    median_latency_ms: float
    p99_latency_ms: float


# Since vllm must support Python 3.8, we can't use str.removeprefix(prefix)
# introduced in Python 3.9
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


class Backend:
    def __init__(self):
        raise NotImplementedError

    def add_cli_arguments(self, parser: argparse.ArgumentParser):
        raise NotImplementedError

    def initialize(self, args):
        raise NotImplementedError

    def async_request(
        self, request: RequestFuncInput, pbar: Optional[tqdm] = None
    ) -> RequestFuncOutput:
        raise NotImplementedError


class OpenAIBackend(Backend):
    def __init__(self):
        self.api_url = None
        self.model_id = None
        self.tokenizer_id = None
        self.tokenizer = None

    def add_cli_arguments(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            "--base-url",
            type=str,
            default=None,
            help="Server or API base url if not using http host and port.",
        )
        parser.add_argument("--host", type=str, default="localhost")
        parser.add_argument("--port", type=int, default=8000)
        parser.add_argument(
            "--endpoint",
            type=str,
            default="/v1/completions",
            help="API endpoint.",
        )
        parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Name of the model.",
        )
        parser.add_argument(
            "--tokenizer",
            type=str,
            help="Name or path of the tokenizer, if not using the default tokenizer.",
        )
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Trust remote code from huggingface",
        )

    def initialize(self, args):
        if args.base_url is not None:
            self.api_url = f"{args.base_url}{args.endpoint}"
        else:
            self.api_url = f"http://{args.host}:{args.port}{args.endpoint}"
        self.model_id = args.model
        self.tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
        self.tokenizer = get_tokenizer(
            self.tokenizer_id, trust_remote_code=args.trust_remote_code
        )

    async def async_request(
        self, request: RequestFuncInput, pbar: Optional[tqdm] = None
    ) -> RequestFuncOutput:
        api_url = request.api_url
        assert api_url.endswith(
            "v1/completions"
        ), "OpenAI Completions API URL must end with 'v1/completions'."

        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            assert not request.use_beam_search
            payload = {
                "model": request.model,
                "prompt": request.prompt,
                "temperature": request.temperature,
                "best_of": request.best_of,
                "min_tokens": request.output_len,
                "max_tokens": request.output_len,
                "ignore_eos": request.ignore_eos,
                "stream": True,
                "stream_options": {
                    "include_usage": True,
                },
            }
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}

            output = RequestFuncOutput()
            output.prompt_len = request.prompt_len
            output.completion_len = 0
            output.generated_text = ""
            output.ttft = 0.0
            output.latency = 0.0

            st = time.perf_counter()
            most_recent_timestamp = st
            try:
                async with session.post(
                    url=api_url,
                    json=payload,
                    headers=headers,
                    raise_for_status=True,
                ) as response:
                    async for chunk_bytes in response.content:
                        chunk = remove_prefix(
                            chunk_bytes.strip().decode("utf-8"),
                            "data: ",
                        )
                        if not chunk or chunk == "[DONE]":
                            continue
                        data = json.loads(chunk)
                        new_completion_len = data["usage"]["completion_tokens"]
                        assert output.completion_len <= new_completion_len
                        if new_completion_len == output.completion_len:
                            continue  # No new tokens
                        timestamp = time.perf_counter()
                        if not output.completion_len:
                            # First token
                            output.ttft = time.perf_counter() - st
                        else:
                            # Decoding phase
                            output.itl.append(
                                timestamp - most_recent_timestamp
                            )
                        most_recent_timestamp = timestamp
                        output.generated_text += data["choices"][0]["text"]
                        output.completion_len = new_completion_len
                    output.latency = time.perf_counter() - st
                    output.success = True
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


class VLLMBackend(Backend):
    def __init__(self):
        self.api_url = None
        self.model_id = None
        self.tokenizer_id = None
        self.tokenizer = None
        self.engine = None

    def add_cli_arguments(self, parser: argparse.ArgumentParser):
        parser = AsyncEngineArgs.add_cli_args(parser)

    def initialize(self, args):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.model_id = args.model
        self.tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model
        self.tokenizer = get_tokenizer(
            self.tokenizer_id, trust_remote_code=args.trust_remote_code
        )

    async def async_request(
        self, request: RequestFuncInput, pbar: Optional[tqdm] = None
    ) -> RequestFuncOutput:
        sampling_params = SamplingParams(
            temperature=request.temperature,
            best_of=request.best_of,
            min_tokens=request.output_len,
            max_tokens=request.output_len,
            ignore_eos=request.ignore_eos,
        )
        request_id = random_uuid()
        output = RequestFuncOutput()
        output.prompt_len = request.prompt_len
        output.completion_len = 0
        output.error = ""
        output.ttft = 0.0

        st = time.perf_counter()
        assert self.engine is not None
        results_generator = self.engine.generate(
            request.prompt, sampling_params, request_id
        )
        try:
            async for request_output in results_generator:
                assert len(request_output.outputs) == 1
                new_completion_len = len(request_output.outputs[0].token_ids)
                assert output.completion_len <= new_completion_len
                if new_completion_len == output.completion_len:
                    continue  # No new tokens
                timestamp = time.perf_counter()
                if output.ttft == 0.0:
                    output.ttft = timestamp - st
                else:
                    output.itl.append(timestamp - most_recent_timestamp)
                most_recent_timestamp = timestamp
                output.completion_len = new_completion_len
            assert request_output is not None
            output.latency = time.perf_counter() - st
            output.success = True
            output.generated_text = request_output.outputs[0].text
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)

        return output


backends = {
    "openai": OpenAIBackend,
    "vllm": VLLMBackend,
}


async def get_request(
    input_requests: List[TraceEntry],
) -> AsyncGenerator[TraceEntry, None]:
    """Get the request generator

    Args:
        input_requests (List[TraceEntry]): A list of requests, with timestamp and list of queries, each containing a prompt, prompt length, and response length.

    Returns:
        AsyncGenerator[TraceEntry, None]: The request generator

    Yields:
        Iterator[AsyncGenerator[TraceEntry, None]]: The request iterator
    """
    input_requests = iter(input_requests)
    last_req_timestamp_ms = 0
    for request in input_requests:
        await asyncio.sleep((request.timestamp - last_req_timestamp_ms) / 1000.0)
        last_req_timestamp_ms = request.timestamp
        yield request


def calculate_metrics(
    input_requests: List[TraceEntry],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens: List[int] = []
    total_input = 0
    completed = 0
    latencies: List[float] = []
    itls: List[float] = []
    tpots: List[float] = []
    ttfts: List[float] = []

    assert len(outputs) == sum(len(req.queries) for req in input_requests)
    req_idx = 0
    query_idx = 0
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = outputs[i].completion_len
            actual_output_lens.append(output_len)
            total_input += input_requests[req_idx].queries[query_idx].prompt_length
            if output_len > 1:
                tpots.append((outputs[i].latency - outputs[i].ttft) / (output_len - 1))
            itls += outputs[i].itl
            ttfts.append(outputs[i].ttft)
            latencies.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

        query_idx += 1
        if query_idx >= len(input_requests[req_idx].queries):
            query_idx = 0
            req_idx += 1
    assert req_idx == len(input_requests) and query_idx == 0

    if completed == 0:
        warnings.warn(
            "All requests failed. This is likely due to a misconfiguration "
            "on the benchmark arguments.",
            stacklevel=2,
        )
    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_ttft_ms=np.mean(ttfts or 0)
        * 1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots or 0) * 1000,
        median_tpot_ms=np.median(tpots or 0) * 1000,
        p99_tpot_ms=np.percentile(tpots or 0, 99) * 1000,
        mean_itl_ms=np.mean(itls or 0) * 1000,
        median_itl_ms=np.median(itls or 0) * 1000,
        p99_itl_ms=np.percentile(itls or 0, 99) * 1000,
        mean_latency_ms=np.mean(latencies or 0) * 1000,
        median_latency_ms=np.median(latencies or 0) * 1000,
        p99_latency_ms=np.percentile(latencies or 0, 99) * 1000,
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: Backend,
    input_requests: List[TraceEntry],
    best_of: int,
    use_beam_search: bool,
    disable_tqdm: bool,
):

    print("Starting initial single prompt test run...")
    test_request = input_requests[0]
    for test_query in test_request.queries:
        test_input = RequestFuncInput(
            model=backend.model_id,
            prompt=test_query.prompt,
            api_url=backend.api_url,
            prompt_len=test_query.prompt_length,
            output_len=test_query.response_length,
            best_of=best_of,
            use_beam_search=use_beam_search,
            ignore_eos=True,
        )
        test_output = await backend.async_request(request=test_input)
        if not test_output.success:
            raise ValueError(
                "Initial test run failed - Please make sure benchmark arguments "
                f"are correctly specified. Error: {test_output.error}"
            )
    print("Initial test run completed. Starting main benchmark run...")

    total_reqs_count = int(sum(len(req.queries) for req in input_requests))
    pbar = None if disable_tqdm else tqdm(total=total_reqs_count)

    benchmark_start_time = time.perf_counter()
    tasks: List[asyncio.Task] = []

    async for request in get_request(input_requests):
        for query in request.queries:
            req = RequestFuncInput(
                model=backend.model_id,
                prompt=query.prompt,
                api_url=backend.api_url,
                prompt_len=query.prompt_length,
                output_len=query.response_length,
                best_of=best_of,
                use_beam_search=use_beam_search,
                ignore_eos=True,
            )
            tasks.append(
                asyncio.create_task(backend.async_request(request=req, pbar=pbar))
            )
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=backend.tokenizer,
    )

    print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):", benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Request throughput (req/s):", metrics.request_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Input token throughput (tok/s):", metrics.input_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print("{s:{c}^{n}}".format(s="Time to First Token", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):", metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print(
        "{s:{c}^{n}}".format(s="Time per Output Token (excl. 1st token)", n=50, c="-")
    )
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):", metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("{s:{c}^{n}}".format(s="Inter-token Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean ITL (ms):", metrics.mean_itl_ms))
    print("{:<40} {:<10.2f}".format("Median ITL (ms):", metrics.median_itl_ms))
    print("{:<40} {:<10.2f}".format("P99 ITL (ms):", metrics.p99_itl_ms))
    print("{s:{c}^{n}}".format(s="Request Latency", n=50, c="-"))
    print("{:<40} {:<10.2f}".format("Mean Latency (ms):", metrics.mean_latency_ms))
    print("{:<40} {:<10.2f}".format("Median Latency (ms):", metrics.median_latency_ms))
    print("{:<40} {:<10.2f}".format("P99 Latency (ms):", metrics.p99_latency_ms))
    print("=" * 50)

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "mean_itl_ms": metrics.mean_itl_ms,
        "median_itl_ms": metrics.median_itl_ms,
        "p99_itl_ms": metrics.p99_itl_ms,
        "mean_latency_ms": metrics.mean_latency_ms,
        "median_latency_ms": metrics.median_latency_ms,
        "p99_latency_ms": metrics.p99_latency_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        "itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }
    return result


def parse_trace_file(trace_file: str) -> List[TraceEntry]:
    trace_entries = []
    with open(trace_file, "r") as file:
        for line in file:
            json_data = json.loads(line)
            trace_entry = TraceEntry.from_dict(json_data)
            trace_entries.append(trace_entry)
    trace_entries.sort(key=lambda entry: entry.timestamp)
    return trace_entries


def generate_prompts(
    trace: List[TraceEntry],
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
) -> List[TraceEntry]:
    def generate_random_prompt(length: int) -> str:
        # Generate random integers
        random_tokens = torch.randint(0, tokenizer.vocab_size, (int(length),))
        # Decode the random tokens
        return tokenizer.decode(random_tokens)

    for entry in tqdm(trace, desc="Generating prompts"):
        for query in entry.queries:
            query.prompt = generate_random_prompt(query.prompt_length)
    return trace


def set_arrival_timestamps(
    trace: List[TraceEntry],
    arrival_type: str,
    arrival_rate: Optional[float],
    arrival_seed: int,
) -> List[TraceEntry]:
    # Ensure first request is always at time 0.
    min_ts = min(r.timestamp for r in trace)
    for req in trace:
        req.timestamp -= min_ts
    if len(trace) <= 1 or arrival_type == "trace" and arrival_rate is None:
        return trace  # Use the same arrivals in the trace.
    old_max_ts = max(r.timestamp for r in trace)
    new_max_ts = (old_max_ts if arrival_rate is None else
                  int((len(trace) - 1) / arrival_rate * 1000))
    if arrival_type == "trace":
        if old_max_ts > 0:
            # Rescale timestamps to match the new maximum timestamp.
            for req in trace:
                req.timestamp = new_max_ts * req.timestamp // old_max_ts
            return trace
        else:
            # If all timestamps are the same, use a uniform distribution
            arrival_type = "uniform"
    sorted_requests = sorted(trace, key=lambda x: x.timestamp)
    if arrival_type == "random":
        # Use a random distribution for arrival times.
        rng = random.Random(arrival_seed)
        samples = sorted([rng.random() for _ in trace])
        min_sample, max_sample = min(samples), max(samples)
        samples = [(s - min_sample) / (max_sample - min_sample) for s in samples]
        for sample, req in zip(samples, sorted_requests):
            req.timestamp = int(sample * new_max_ts)
    elif arrival_type == "uniform":
        # Use a uniform distribution for arrival times.
        for i, req in enumerate(sorted_requests):
            req.timestamp = i * new_max_ts // (len(trace) - 1)
    else:
        raise ValueError(f"Unknown arrival type: {arrival_type}")
    return trace


def set_request_lengths(
    trace: List[TraceEntry],
    input_length: Optional[int] = None,
    output_length: Optional[int] = None,
) -> List[TraceEntry]:
    for req in trace:
        for query in req.queries:
            if input_length is not None:
                query.prompt_length = input_length
            if output_length is not None:
                query.response_length = output_length
    return trace


def main(args: argparse.Namespace):

    backend_class = backends[args.backend]
    backend = backend_class()
    backend.add_cli_arguments(parser)
    args = parser.parse_args()

    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend.initialize(args)

    # Load trace data
    trace = parse_trace_file(trace_file=args.trace_file)

    # Set arrival timestamps
    trace = set_arrival_timestamps(
        trace=trace,
        arrival_type=args.arrival_type,
        arrival_rate=args.arrival_rate,
        arrival_seed=args.arrival_seed,
    )

    # Override request lengths
    trace = set_request_lengths(
        trace=trace,
        input_length=args.input_length,
        output_length=args.output_length,
    )

    if args.max_requests >= 0 and len(trace) >= args.max_requests:
        trace = trace[:args.max_requests]

    trace = generate_prompts(trace=trace, tokenizer=backend.tokenizer)

    # Replay Trace on local VLLM server
    benchmark_result = asyncio.get_event_loop().run_until_complete(
        benchmark(
            backend=backend,
            input_requests=trace,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            disable_tqdm=args.disable_tqdm,
        )
    )

    # Save config and results to json
    if args.save_result:
        result_json: Dict[str, Any] = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["model_id"] = backend.model_id
        result_json["tokenizer_id"] = backend.tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["max_requests"] = args.max_requests

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = backend.model_id.split("/")[-1]
        file_name = f"vllm-{base_model_id}-{current_dt}.json"  # noqa
        if args.result_filename:
            file_name = args.result_filename
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile)


def add_trace_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--trace-file", type=str, required=True,
        help="Path to a trace file in .jsonl format.",
    )
    parser.add_argument(
        "--arrival-type", type=str, default="trace",
        choices=["trace", "random", "uniform"],
        help=("Type of arrival rate. 'trace' uses the arrivals specified in "
              "the trace file. 'random' uses a random distribution, and "
              "'uniform' uses a uniform distribution (evenly spaced)."),
    )
    parser.add_argument(
        "--arrival-rate", type=float,
        help=("Arrival rate of requests in requests per second. If not "
              "specified, the arrival rate is determined by the trace."),
    )
    parser.add_argument(
        "--arrival-seed", type=int, default=42,
        help="Seed for generating arrival times.",
    )
    parser.add_argument(
        "--input-length", type=int,
        help="Override the input lengths of the requests.",
    )
    parser.add_argument(
        "--output-length", type=int,
        help="Override the output lengths of the requests.",
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser = add_trace_args(parser)
    parser.add_argument(
        "--backend",
        type=str,
        choices=backends.keys(),
        required=True,
        help="Backend to benchmark.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--max-requests",
        type=int,
        default=-1,
        help="Number of requests to process.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )
    parser.add_argument(
        "--result-filename",
        type=str,
        default=None,
        help="Specify the filename to save benchmark json results."
        "If not specified, results will be saved in "
        "vllm-{base_model_id}-{current_dt}.json"
        " format.",
    )

    args, _ = parser.parse_known_args()
    main(args)
