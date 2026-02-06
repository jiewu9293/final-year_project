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
from clients.openai_client import OpenAIClient
from utils.io import append_jsonl


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
        choices=["zero_shot"],
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
        default=0.2,
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
        help="Number of test generation rounds (currently only k=1 supported)"
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
        default="results/benchmark_results.jsonl",
        help="JSONL file to append results"
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
    
    if args.k != 1:
        print(f"Warning: k={args.k} specified, but only k=1 is currently supported. Using k=1.")
        args.k = 1
    
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
    
    framework = ZeroShotPrompting()
    print(f"Using framework: {args.framework}")
    
    client = OpenAIClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    print(f"Using model: {args.model}")
    
    out_base = Path(args.out_dir) / args.benchmark / args.framework / args.model
    out_base.mkdir(parents=True, exist_ok=True)
    
    results_file = Path(args.results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {out_base}")
    print(f"Results file: {results_file}")
    print(f"\nStarting generation (k={args.k})...\n")
    
    for idx, item in enumerate(tqdm(dataset, desc="Generating tests")):
        try:
            task = item_to_task(item, benchmark_name=args.benchmark)
            
            bundle = framework.build(task)
            
            start_time = time.time()
            generated_code = client.generate(bundle.to_openai_like())
            generated_code = clean_markdown_code_blocks(generated_code)
            latency = time.time() - start_time
            
            task_out_dir = out_base / f"task_{item['task_id']}"
            task_out_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = task_out_dir / f"tests_k{args.k}.py"
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
                'temperature': args.temperature,
                'max_tokens': args.max_tokens,
                'latency': latency,
                'input_tokens': client.last_input_tokens,
                'output_tokens': client.last_output_tokens,
                'output_file': str(output_file),
                'status': 'success',
                'error': None
            }
            
            append_jsonl(results_file, record)
            
        except Exception as e:
            print(f"\nError processing task {item.get('task_id', idx)}: {e}")
            
            record = {
                'timestamp': time.time(),
                'benchmark': args.benchmark,
                'task_id': f"{args.benchmark}_{item.get('task_id', idx)}",
                'original_task_id': item.get('task_id', idx),
                'func_name': item.get('func_name', ''),
                'framework': args.framework,
                'model': args.model,
                'k': args.k,
                'status': 'error',
                'error': str(e)
            }
            
            append_jsonl(results_file, record)
            continue


if __name__ == "__main__":
    main()
