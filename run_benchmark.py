"""
Batch runner for benchmark test generation.
Supports UnLeakedTestBench (ULT/PLT) and other datasets.
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from benchmarks_adapter import load_json_array_or_jsonl, item_to_task
from prompting.zero_shot import ZeroShotPrompting
from prompting.few_shot import FewShotPrompting
from prompting.cot import CoTPrompting
from prompting.gtot import GToTPrompting
from prompting.tot import ToTPrompting
from clients.openai_client import OpenAIClient
from clients.anthropic_client import AnthropicClient
from clients.deepseek_client import DeepSeekClient
from clients.google_client import GoogleClient
from utils.io import append_jsonl


# ── Supported models (11 total) ────────────────────────────────────────
SUPPORTED_MODELS = [
    # OpenAI (High/Mid/Low)
    "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano",
    # Anthropic (High/Mid/Low)
    "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5",
    # Google (High/Mid/Low)
    "gemini-3.1-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview",
    # DeepSeek (Reasoning/General)
    "deepseek-reasoner", "deepseek-chat",
]


def get_client(model: str, temperature: float, max_tokens: int):
    """Select the appropriate LLM client based on model name prefix."""
    if model.startswith("claude-"):
        return AnthropicClient(model=model, temperature=temperature, max_tokens=max_tokens)
    elif model.startswith("gemini-"):
        return GoogleClient(model=model, temperature=temperature, max_tokens=max_tokens)
    elif model.startswith("deepseek-"):
        return DeepSeekClient(model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        return OpenAIClient(model=model, temperature=temperature, max_tokens=max_tokens)


def parse_args():
    parser = argparse.ArgumentParser(description="Run benchmark test generation")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset file (e.g., benchmarks/UnLeakedTestBench/datasets/ULT.jsonl)"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="ult",
        help="Benchmark name (ult, plt, ult_lite)"
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "few_shot", "cot", "tot", "gtot"],
        help="Prompt framework to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model name (e.g., gpt-4o, gpt-4-turbo)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Number of independent samples to generate per task"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to process"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N tasks"
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated task IDs to process (e.g., '1,4,7')"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="Output directory for generated tests"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="JSONL file to append results (default: results/{framework}/{model}/k{k}/benchmark_results.jsonl)"
    )
    
    return parser.parse_args()

def clean_markdown_code_blocks(text: str) -> str:
    """Remove markdown code block markers from generated code."""
    import re
    # Remove ```python or ``` at start/end
    text = re.sub(r'^```python\s*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^```\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n```\s*$', '', text)
    return text.strip()

def main():
    args = parse_args()
    
    print(f"Loading dataset from: {args.dataset}")
    task_ids_list = None
    if args.task_ids:
        task_ids_list = [tid.strip() for tid in args.task_ids.split(',')]
    
    dataset = load_json_array_or_jsonl(
        args.dataset,
        limit=args.limit,
        offset=args.offset,
        task_ids=task_ids_list
    )
    print(f"Loaded {len(dataset)} tasks")
    
    FRAMEWORKS = {
        'zero_shot': ZeroShotPrompting,
        'few_shot': FewShotPrompting,
        'cot': CoTPrompting,
        'tot': ToTPrompting,
        'gtot': GToTPrompting,
    }
    framework = FRAMEWORKS[args.framework]()
    print(f"Using framework: {args.framework}")
    
    client = get_client(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(f"Using model: {args.model}")
    
    out_base = Path(args.out_dir) / args.benchmark / args.framework / args.model / f"k{args.k}"
    out_base.mkdir(parents=True, exist_ok=True)
    
    results_file = Path(args.results_file) if args.results_file else Path(f"results/{args.framework}/{args.model}/k{args.k}/benchmark_results.jsonl")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {out_base}")
    print(f"Results file: {results_file}")
    print(f"\nStarting generation (k={args.k})...\n")
    
    for idx, item in enumerate(tqdm(dataset, desc="Generating tests")):
        task_out_dir = out_base / f"task_{item['task_id']}"
        task_out_dir.mkdir(parents=True, exist_ok=True)
        
        for sample_idx in range(1, args.k + 1):
            try:
                task = item_to_task(item, benchmark_name=args.benchmark)
                
                start_time = time.time()
                
                # Check if framework supports multi-step execution
                if hasattr(framework, 'run_multistep'):
                    # GToT: multi-step execution (3 LLM calls)
                    bundle = framework.run_multistep(task, client)
                    generated_code = bundle.extra.get('final_response', '')
                else:
                    # Zero-shot/Few-shot: single-step execution
                    bundle = framework.build(task)
                    generated_code = client.generate(bundle.to_openai_like())
                
                generated_code = clean_markdown_code_blocks(generated_code)
                latency = time.time() - start_time
                
                output_file = task_out_dir / f"tests_s{sample_idx}.py"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(generated_code)
                
                record = {
                    'timestamp': time.time(),
                    'benchmark': args.benchmark,
                    'task_id': task.task_id,
                    'original_task_id': item['task_id'],
                    'func_name': item.get('func_name', ''),
                    'framework': args.framework,
                    'model': args.model,
                    'k': args.k,
                    'sample': sample_idx,
                    'temperature': args.temperature,
                    'max_tokens': args.max_tokens,
                    'latency': latency,
                    'input_tokens': client.last_input_tokens,
                    'output_tokens': client.last_output_tokens,
                    'energy_kwh_min': client.last_energy_kwh[0] if client.last_energy_kwh else None,
                    'energy_kwh_max': client.last_energy_kwh[1] if client.last_energy_kwh else None,
                    'gwp_kgco2eq_min': client.last_gwp_kgco2eq[0] if client.last_gwp_kgco2eq else None,
                    'gwp_kgco2eq_max': client.last_gwp_kgco2eq[1] if client.last_gwp_kgco2eq else None,
                    'adpe_kgsbeq_min': client.last_adpe_kgsbeq[0] if client.last_adpe_kgsbeq else None,
                    'adpe_kgsbeq_max': client.last_adpe_kgsbeq[1] if client.last_adpe_kgsbeq else None,
                    'pe_mj_min': client.last_pe_mj[0] if client.last_pe_mj else None,
                    'pe_mj_max': client.last_pe_mj[1] if client.last_pe_mj else None,
                    'output_file': str(output_file),
                    'status': 'success',
                    'error': None
                }
                
                append_jsonl(results_file, record)
                
            except Exception as e:
                print(f"\nError processing task {item.get('task_id', idx)} sample {sample_idx}: {e}")
                
                record = {
                    'timestamp': time.time(),
                    'benchmark': args.benchmark,
                    'task_id': f"{args.benchmark}_{item.get('task_id', idx)}",
                    'original_task_id': item.get('task_id', idx),
                    'func_name': item.get('func_name', ''),
                    'framework': args.framework,
                    'model': args.model,
                    'k': args.k,
                    'sample': sample_idx,
                    'status': 'error',
                    'error': str(e)
                }
                
                append_jsonl(results_file, record)
                continue


if __name__ == "__main__":
    main()
