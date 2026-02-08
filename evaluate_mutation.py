"""
Evaluate generated tests using mutation testing with mutmut.
Calculates Mutation Score (Mut@k) for each task.
"""

import argparse
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from benchmarks_adapter import load_json_array_or_jsonl
from utils.io import append_jsonl


CODE_IMPORTS = """
import os
import re
import math
import numpy
import pandas
import pytest
from typing import *
from collections import *
from itertools import *
from functools import *
"""


def run_mutmut_on_task(
    task_id: str,
    original_code: str,
    generated_tests: str,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Run mutmut mutation testing on a single task.
    
    Args:
        task_id: Task identifier
        original_code: Source code of the function under test
        generated_tests: Generated test code
        timeout: Timeout in seconds (mutation testing is slow)
    
    Returns:
        Dictionary with mutation testing results
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mod.py (code under test)
            mod_file = temp_path / "mod.py"
            with open(mod_file, 'w', encoding='utf-8') as f:
                f.write(CODE_IMPORTS)
                f.write("\n\n")
                f.write(original_code)
            
            # Create test_mod.py (tests)
            test_file = temp_path / "test_mod.py"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("from mod import *\n\n")
                f.write(generated_tests)
            
            # Run mutmut
            result = subprocess.run(
                ['mutmut', 'run', '--paths-to-mutate=mod.py', '--no-progress'],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Get results summary
            result_summary = subprocess.run(
                ['mutmut', 'results'],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Parse mutmut output
            mutation_data = parse_mutmut_output(result_summary.stdout)
            
            return {
                'task_id': task_id,
                'status': 'success',
                **mutation_data,
                'stdout': result_summary.stdout[-1000:] if len(result_summary.stdout) > 1000 else result_summary.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            'task_id': task_id,
            'status': 'timeout',
            'error': f'Mutation testing timeout after {timeout}s'
        }
    except Exception as e:
        return {
            'task_id': task_id,
            'status': 'error',
            'error': str(e)
        }


def parse_mutmut_output(stdout: str) -> Dict[str, Any]:
    """
    Parse mutmut results output.
    
    Expected format:
    Survived: 3
    Killed: 12
    Timeout: 0
    Suspicious: 0
    
    Returns:
        {
            'mutations_killed': int,
            'mutations_survived': int,
            'mutations_timeout': int,
            'mutations_total': int,
            'mutation_score': float
        }
    """
    import re
    
    killed = 0
    survived = 0
    timeout = 0
    suspicious = 0
    
    # Parse output
    killed_match = re.search(r'Killed[:\s]+(\d+)', stdout, re.IGNORECASE)
    if killed_match:
        killed = int(killed_match.group(1))
    
    survived_match = re.search(r'Survived[:\s]+(\d+)', stdout, re.IGNORECASE)
    if survived_match:
        survived = int(survived_match.group(1))
    
    timeout_match = re.search(r'Timeout[:\s]+(\d+)', stdout, re.IGNORECASE)
    if timeout_match:
        timeout = int(timeout_match.group(1))
    
    suspicious_match = re.search(r'Suspicious[:\s]+(\d+)', stdout, re.IGNORECASE)
    if suspicious_match:
        suspicious = int(suspicious_match.group(1))
    
    total = killed + survived + timeout + suspicious
    mutation_score = killed / total if total > 0 else 0.0
    
    return {
        'mutations_killed': killed,
        'mutations_survived': survived,
        'mutations_timeout': timeout,
        'mutations_suspicious': suspicious,
        'mutations_total': total,
        'mutation_score': mutation_score
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated tests with mutation testing")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to original dataset"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results/benchmark_results.jsonl",
        help="Generation results file to read"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="results/mutation_results.jsonl",
        help="Output file for mutation testing results"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to evaluate"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per task in seconds (mutation testing is slow)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"Loading dataset from: {args.dataset}")
    dataset = load_json_array_or_jsonl(args.dataset)
    dataset_dict = {item['task_id']: item for item in dataset}
    print(f"Loaded {len(dataset_dict)} tasks from dataset")
    
    print(f"Loading generation results from: {args.results_file}")
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return
    
    generation_results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                generation_results.append(json.loads(line))
    
    successful_results = [
        r for r in generation_results 
        if r.get('status') == 'success' and Path(r.get('output_file', '')).exists()
    ]
    
    print(f"Found {len(successful_results)} successful generations to evaluate")
    
    if args.limit:
        successful_results = successful_results[:args.limit]
        print(f"Limiting to {args.limit} tasks")
    
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Output file: {output_file}")
    print(f"Starting mutation testing...\n")
    
    total_killed = 0
    total_survived = 0
    total_mutations = 0
    evaluation_success = 0
    evaluation_errors = 0
    
    for gen_result in tqdm(successful_results, desc="Running mutation tests"):
        try:
            original_task_id = gen_result['original_task_id']
            
            if original_task_id not in dataset_dict:
                print(f"\nWarning: task_id {original_task_id} not found in dataset")
                continue
            
            original_item = dataset_dict[original_task_id]
            original_code = original_item['code']
            
            test_file_path = Path(gen_result['output_file'])
            with open(test_file_path, 'r', encoding='utf-8') as f:
                generated_tests = f.read()
            
            mutation_result = run_mutmut_on_task(
                task_id=gen_result['task_id'],
                original_code=original_code,
                generated_tests=generated_tests,
                timeout=args.timeout
            )
            
            mutation_result.update({
                'original_task_id': original_task_id,
                'func_name': gen_result.get('func_name', ''),
                'model': gen_result.get('model', ''),
                'framework': gen_result.get('framework', ''),
                'output_file': str(test_file_path)
            })
            
            if mutation_result['status'] == 'success':
                evaluation_success += 1
                total_killed += mutation_result['mutations_killed']
                total_survived += mutation_result['mutations_survived']
                total_mutations += mutation_result['mutations_total']
            else:
                evaluation_errors += 1
            
            append_jsonl(output_file, mutation_result)
            
        except Exception as e:
            print(f"\nError evaluating {gen_result.get('task_id', 'unknown')}: {e}")
            evaluation_errors += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"Mutation Testing Complete!")
    print(f"{'='*60}")
    print(f"Tasks evaluated: {evaluation_success + evaluation_errors}")
    print(f"  - Successful: {evaluation_success}")
    print(f"  - Errors: {evaluation_errors}")
    print(f"\nMutation Testing Results:")
    print(f"  - Total mutations: {total_mutations}")
    print(f"  - Killed: {total_killed}")
    print(f"  - Survived: {total_survived}")
    if total_mutations > 0:
        overall_mutation_score = total_killed / total_mutations
        print(f"  - Mutation Score (Mut@k): {overall_mutation_score:.2%}")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
