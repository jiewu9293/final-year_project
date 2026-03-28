"""
Batch runner: automatically runs all model × framework × k combinations.
Calls run_benchmark.py for each configuration.

Usage:
    python run_all_experiments.py                            # run all (default)
    python run_all_experiments.py --limit 20                 # 20 tasks per config
    python run_all_experiments.py --k-values 1 3             # only k=1 and k=3
    python run_all_experiments.py --frameworks cot gtot      # only these frameworks
    python run_all_experiments.py --models gpt-5.4 deepseek-chat  # specific models
    python run_all_experiments.py --dry-run                  # preview without running
"""

import argparse
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from tqdm import tqdm

# ── Default experiment configuration ───────────────────────────────────────

DEFAULT_MODELS = [
    # OpenAI
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    # Anthropic
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    # DeepSeek
    "deepseek-reasoner",
    "deepseek-chat",
    # Google (commented out - add back when API key is available)
    # "gemini-3.1-pro-preview",
    # "gemini-3-flash-preview",
    # "gemini-3.1-flash-lite-preview",
]

DEFAULT_FRAMEWORKS = ["zero_shot", "few_shot", "cot", "tot", "gtot"]

DEFAULT_K_VALUES = [1, 3, 5]

DEFAULT_DATASET = "benchmarks/UnLeakedTestBench/datasets/ULT.jsonl"


# ── Helpers ─────────────────────────────────────────────────────────────────

def results_exist(framework: str, model: str, k: int, results_dir: str) -> bool:
    """Check if benchmark_results.jsonl already exists for this configuration."""
    path = Path(results_dir) / framework / model / f"k{k}" / "benchmark_results.jsonl"
    return path.exists() and path.stat().st_size > 0


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def run_single(framework: str, model: str, k: int, args) -> bool:
    """Run a single benchmark configuration. Returns True on success."""
    cmd = [
        sys.executable, "run_benchmark.py",
        "--dataset", args.dataset,
        "--benchmark", args.benchmark,
        "--framework", framework,
        "--model", model,
        "--k", str(k),
        "--temperature", str(args.temperature),
        "--max-tokens", str(args.max_tokens),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.offset:
        cmd += ["--offset", str(args.offset)]
    if args.results_dir:
        cmd += ["--results-dir", args.results_dir]

    tqdm.write(f"\n{'─'*60}")
    tqdm.write(f"  Framework : {framework}")
    tqdm.write(f"  Model     : {model}")
    tqdm.write(f"  k         : {k}")
    tqdm.write(f"{'─'*60}")

    # Redirect child output to a log file to keep progress bar clean
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{framework}_{model.replace('/', '-')}_k{k}.log"

    start = time.perf_counter()
    try:
        with open(log_file, "w") as lf:
            result = subprocess.run(
                cmd,
                cwd=Path(__file__).parent,
                stdout=lf,
                stderr=lf,
                timeout=args.task_timeout * (args.limit or 100) * k + 300,
            )
        elapsed = time.perf_counter() - start
        if result.returncode == 0:
            tqdm.write(f"  ✅ Done in {format_duration(elapsed)}  (log: {log_file})")
            return True
        else:
            tqdm.write(f"  ❌ Failed after {format_duration(elapsed)}  (see: {log_file})")
            return False
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        tqdm.write(f"  ⏰ Timed out after {format_duration(elapsed)}")
        return False
    except Exception as e:
        tqdm.write(f"  ❌ Error: {e}")
        return False


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run all benchmark experiments")

    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Path to dataset JSONL file")
    parser.add_argument("--benchmark", default="ult",
                        help="Benchmark name (ult, plt)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Models to run (default: all non-Google models)")
    parser.add_argument("--frameworks", nargs="+", default=DEFAULT_FRAMEWORKS,
                        help="Frameworks to run")
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES,
                        help="k values to run (default: 1 3 5)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit tasks per config (None = all tasks)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip first N tasks")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens per generation")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Results directory (default: results/)")
    parser.add_argument("--task-timeout", type=int, default=120,
                        help="Per-task timeout in seconds (used to compute subprocess timeout)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip configs that already have results (default: True)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="Re-run all configs even if results exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print all planned configs without running")

    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    results_dir = args.results_dir or "results"

    combinations = list(product(args.frameworks, args.models, args.k_values))
    total = len(combinations)

    print(f"\n{'='*60}")
    print(f"  Batch Experiment Runner")
    print(f"{'='*60}")
    print(f"  Models     : {len(args.models)}")
    print(f"  Frameworks : {len(args.frameworks)}")
    print(f"  k values   : {args.k_values}")
    print(f"  Total      : {total} configurations")
    print(f"  Limit      : {args.limit or 'all tasks'} tasks per config")
    print(f"  Skip exist : {args.skip_existing}")
    if args.dry_run:
        print(f"\n  [DRY RUN - no actual execution]\n")
    print(f"{'='*60}")

    if args.dry_run:
        for i, (fw, model, k) in enumerate(combinations, 1):
            exists = results_exist(fw, model, k, results_dir)
            status = "⏭  (exists)" if exists else "🔲 (pending)"
            print(f"  {i:3d}/{total}  {status}  {fw:12s}  {model:30s}  k={k}")
        return

    passed, failed, skipped = [], [], []
    overall_start = time.perf_counter()

    # Progress bar (force=True ensures it shows even in non-TTY environments)
    pbar = tqdm(combinations, desc="Overall Progress", unit="config", ncols=100, file=sys.stdout, dynamic_ncols=True)
    
    for i, (fw, model, k) in enumerate(pbar, 1):
        # Update progress bar description
        pbar.set_description(f"[{i}/{total}] {fw}/{model}/k={k}")

        if args.skip_existing and results_exist(fw, model, k, results_dir):
            tqdm.write(f"  ⏭  Skipping (results already exist)")
            skipped.append((fw, model, k))
            continue

        ok = run_single(fw, model, k, args)
        if ok:
            passed.append((fw, model, k))
        else:
            failed.append((fw, model, k))
        
        # Update progress bar postfix with stats
        pbar.set_postfix({"✅": len(passed), "❌": len(failed), "⏭": len(skipped)})
    
    pbar.close()

    elapsed = time.perf_counter() - overall_start

    print(f"\n{'='*60}")
    print(f"  Batch Complete in {format_duration(elapsed)}")
    print(f"{'='*60}")
    print(f"  ✅ Passed  : {len(passed)}")
    print(f"  ❌ Failed  : {len(failed)}")
    print(f"  ⏭  Skipped : {len(skipped)}")

    if failed:
        print(f"\n  Failed configurations:")
        for fw, model, k in failed:
            print(f"    - {fw} / {model} / k={k}")

    print(f"\n  Next step: python evaluate_all.py")


if __name__ == "__main__":
    main()
