"""
Evaluate generated tests by running pytest and mutation testing.
Reads generated tests from outputs/ and runs them against the original code.
Calculates Pass@1, LCov@k, BCov@k, and Mut@k.
"""

import argparse
import json
import subprocess
import tempfile
import re
from pathlib import Path
from typing import Dict, Any, Optional
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


def parse_pytest_output(stdout: str) -> Dict[str, Any]:
    """
    Parse pytest output to extract test results.
    
    Returns:
        {
            'passed': int,
            'failed': int,
            'errors': int,
            'total': int,
            'pass_rate': float
        }
    """
    passed = failed = errors = 0
    
    passed_match = re.search(r'(\d+) passed', stdout)
    if passed_match:
        passed = int(passed_match.group(1))
    
    failed_match = re.search(r'(\d+) failed', stdout)
    if failed_match:
        failed = int(failed_match.group(1))
    
    error_match = re.search(r'(\d+) error', stdout)
    if error_match:
        errors = int(error_match.group(1))
    
    total = passed + failed + errors
    pass_rate = passed / total if total > 0 else 0.0
    
    return {
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'total': total,
        'pass_rate': pass_rate
    }


def parse_coverage_output(stdout: str) -> Dict[str, Any]:
    """
    Parse pytest-cov output to extract coverage metrics.
    
    Expected format:
    Name     Stmts   Miss Branch BrPart  Cover
    ------------------------------------------
    mod.py      53      0     20      1    99%
    
    Returns:
        {
            'line_coverage': float,
            'branch_coverage': float,
            'lines_covered': int,
            'lines_total': int,
            'branches_covered': int,
            'branches_total': int
        }
    """
    coverage_data = {
        'line_coverage': 0.0,
        'branch_coverage': 0.0,
        'lines_covered': 0,
        'lines_total': 0,
        'branches_covered': 0,
        'branches_total': 0
    }
    
    cov_match = re.search(r'mod\.py\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)%', stdout)
    if cov_match:
        stmts = int(cov_match.group(1))
        miss = int(cov_match.group(2))
        branches = int(cov_match.group(3))
        brpart = int(cov_match.group(4))
        total_cov = int(cov_match.group(5))
        
        coverage_data['lines_total'] = stmts
        coverage_data['lines_covered'] = stmts - miss
        coverage_data['line_coverage'] = (stmts - miss) / stmts if stmts > 0 else 0.0
        
        coverage_data['branches_total'] = branches
        coverage_data['branches_covered'] = branches - brpart
        coverage_data['branch_coverage'] = (branches - brpart) / branches if branches > 0 else 0.0
    
    return coverage_data


def parse_mutmut_output(stdout: str) -> Dict[str, Any]:
    """
    Parse mutmut results output.
    
    Expected format (mutmut v3.x):
        mod.x_add__mutmut_1: killed
        mod.x_add__mutmut_2: survived
    """
    killed = 0
    survived = 0
    timeout = 0
    suspicious = 0
    
    for line in stdout.strip().splitlines():
        line = line.strip()
        if ': killed' in line:
            killed += 1
        elif ': survived' in line:
            survived += 1
        elif ': timeout' in line:
            timeout += 1
        elif ': suspicious' in line:
            suspicious += 1
    
    total = killed + survived + timeout + suspicious
    mutation_score = killed / total if total > 0 else 0.0
    
    return {
        'mutations_killed': killed,
        'mutations_survived': survived,
        'mutations_timeout': timeout,
        'mutations_total': total,
        'mutation_score': mutation_score
    }


def run_mutmut_on_task(
    task_id: str,
    original_code: str,
    generated_tests: str,
    timeout: int = 300
) -> Dict[str, Any]:
    """
    Run mutmut mutation testing on a single task.
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            mod_file = temp_path / "mod.py"
            with open(mod_file, 'w', encoding='utf-8') as f:
                f.write(CODE_IMPORTS)
                f.write("\n\n")
                f.write(original_code)
            
            test_file = temp_path / "test_mod.py"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("from mod import *\n\n")
                f.write(generated_tests)
            
            setup_cfg = temp_path / "setup.cfg"
            with open(setup_cfg, 'w', encoding='utf-8') as f:
                f.write("[mutmut]\npaths_to_mutate=mod.py\ntests_dir=.\n")
            
            subprocess.run(
                ['mutmut', 'run'],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            result_summary = subprocess.run(
                ['mutmut', 'results', '--all', 'true'],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return parse_mutmut_output(result_summary.stdout)
            
    except subprocess.TimeoutExpired:
        return {
            'mutations_killed': 0, 'mutations_survived': 0,
            'mutations_timeout': 0, 'mutations_total': 0,
            'mutation_score': 0.0, 'mutation_error': f'timeout after {timeout}s'
        }
    except Exception as e:
        return {
            'mutations_killed': 0, 'mutations_survived': 0,
            'mutations_timeout': 0, 'mutations_total': 0,
            'mutation_score': 0.0, 'mutation_error': str(e)
        }


def run_pytest_on_task(
    task_id: str,
    original_code: str,
    generated_tests: str,
    timeout: int = 30
) -> Dict[str, Any]:
    """
    Run pytest on a single task in an isolated temporary directory.
    
    Args:
        task_id: Task identifier
        original_code: Source code of the function under test
        generated_tests: Generated test code
        timeout: Timeout in seconds
    
    Returns:
        Dictionary with test results or error information
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            mod_file = temp_path / "mod.py"
            with open(mod_file, 'w', encoding='utf-8') as f:
                f.write(CODE_IMPORTS)
                f.write("\n\n")
                f.write(original_code)
            
            test_file = temp_path / "test.py"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write("from mod import *\n\n")
                f.write(generated_tests)
            
            result = subprocess.run(
                ['pytest', str(test_file), '-v', '--tb=short', 
                 '--cov=mod', '--cov-branch', '--cov-report=term'],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            parsed = parse_pytest_output(result.stdout + result.stderr)
            coverage = parse_coverage_output(result.stdout + result.stderr)
            
            return {
                'task_id': task_id,
                'status': 'success',
                'pytest_returncode': result.returncode,
                **parsed,
                **coverage,
                'stdout': result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
                'stderr': result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
            }
            
    except subprocess.TimeoutExpired:
        return {
            'task_id': task_id,
            'status': 'timeout',
            'error': f'Pytest timeout after {timeout}s'
        }
    except Exception as e:
        return {
            'task_id': task_id,
            'status': 'error',
            'error': str(e)
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated tests with pytest")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to original dataset (e.g., benchmarks/UnLeakedTestBench/datasets/ULT.jsonl)"
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
        default="results/evaluation_results.jsonl",
        help="Output file for evaluation results"
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
        default=30,
        help="Timeout per task in seconds for pytest"
    )
    parser.add_argument(
        "--mutation",
        action="store_true",
        help="Also run mutation testing (slow, ~30s per task)"
    )
    parser.add_argument(
        "--mutation-timeout",
        type=int,
        default=300,
        help="Timeout per task in seconds for mutation testing"
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
    if args.mutation:
        print(f"Mutation testing: ENABLED (slow, ~30s per task)")
    print(f"\nStarting evaluation...\n")
    
    total_passed = 0
    total_failed = 0
    total_errors = 0
    evaluation_success = 0
    evaluation_errors = 0
    total_line_coverage = 0.0
    total_branch_coverage = 0.0
    coverage_count = 0
    total_mutation_score = 0.0
    mutation_count = 0
    total_mutations_killed = 0
    total_mutations_total = 0
    
    for gen_result in tqdm(successful_results, desc="Evaluating tests"):
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
            
            eval_result = run_pytest_on_task(
                task_id=gen_result['task_id'],
                original_code=original_code,
                generated_tests=generated_tests,
                timeout=args.timeout
            )
            
            eval_result.update({
                'original_task_id': original_task_id,
                'func_name': gen_result.get('func_name', ''),
                'model': gen_result.get('model', ''),
                'framework': gen_result.get('framework', ''),
                'output_file': str(test_file_path)
            })
            
            if args.mutation and eval_result.get('status') == 'success':
                mutation_data = run_mutmut_on_task(
                    task_id=gen_result['task_id'],
                    original_code=original_code,
                    generated_tests=generated_tests,
                    timeout=args.mutation_timeout
                )
                eval_result.update(mutation_data)
            
            if eval_result['status'] == 'success':
                evaluation_success += 1
                total_passed += eval_result['passed']
                total_failed += eval_result['failed']
                total_errors += eval_result['errors']
                if 'line_coverage' in eval_result:
                    total_line_coverage += eval_result['line_coverage']
                    total_branch_coverage += eval_result['branch_coverage']
                    coverage_count += 1
                if eval_result.get('mutations_total', 0) > 0:
                    total_mutation_score += eval_result['mutation_score']
                    total_mutations_killed += eval_result['mutations_killed']
                    total_mutations_total += eval_result['mutations_total']
                    mutation_count += 1
            else:
                evaluation_errors += 1
            
            append_jsonl(output_file, eval_result)
            
        except Exception as e:
            print(f"\nError evaluating {gen_result.get('task_id', 'unknown')}: {e}")
            evaluation_errors += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}")
    print(f"Tasks evaluated: {evaluation_success + evaluation_errors}")
    print(f"  - Successful: {evaluation_success}")
    print(f"  - Errors: {evaluation_errors}")
    print(f"\nTest Results:")
    print(f"  - Total tests: {total_passed + total_failed + total_errors}")
    print(f"  - Passed: {total_passed}")
    print(f"  - Failed: {total_failed}")
    print(f"  - Errors: {total_errors}")
    if total_passed + total_failed + total_errors > 0:
        overall_pass_rate = total_passed / (total_passed + total_failed + total_errors)
        print(f"  - Overall pass rate: {overall_pass_rate:.2%}")
    
    if coverage_count > 0:
        avg_line_cov = total_line_coverage / coverage_count
        avg_branch_cov = total_branch_coverage / coverage_count
        print(f"\nCoverage Metrics:")
        print(f"  - Average Line Coverage (LCov@k): {avg_line_cov:.2%}")
        print(f"  - Average Branch Coverage (BCov@k): {avg_branch_cov:.2%}")
    
    if mutation_count > 0:
        avg_mutation_score = total_mutation_score / mutation_count
        overall_mutation_score = total_mutations_killed / total_mutations_total if total_mutations_total > 0 else 0.0
        print(f"\nMutation Testing Metrics:")
        print(f"  - Tasks with mutations: {mutation_count}")
        print(f"  - Total mutations: {total_mutations_total}")
        print(f"  - Killed: {total_mutations_killed}")
        print(f"  - Mutation Score (Mut@k): {overall_mutation_score:.2%}")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
