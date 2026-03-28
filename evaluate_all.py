"""
Batch evaluator: automatically evaluates all completed benchmark configurations.
Calls evaluate_tests.py for each configuration that has benchmark_results.jsonl.

Usage:
    python evaluate_all.py                          # evaluate all completed configs
    python evaluate_all.py --mutation               # include mutation testing (slow)
    python evaluate_all.py --frameworks cot gtot    # only specific frameworks
    python evaluate_all.py --models gpt-5.4         # only specific models
    python evaluate_all.py --k-values 1             # only k=1
    python evaluate_all.py --dry-run                # preview without running
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from tqdm import tqdm

DEFAULT_DATASET = "benchmarks/UnLeakedTestBench/datasets/ULT.jsonl"


# ── Helpers ──────────────────────────────────────────────────────────────────

def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def discover_configs(results_dir: str, frameworks=None, models=None, k_values=None):
    """
    Scan results/ directory for benchmark_results.jsonl files.
    Returns list of (framework, model, k) tuples.
    """
    base = Path(results_dir)
    configs = []

    if not base.exists():
        return configs

    for fw_dir in sorted(base.glob("*/")):
        if not fw_dir.is_dir():
            continue
        fw = fw_dir.name
        if frameworks and fw not in frameworks:
            continue

        for model_dir in sorted(fw_dir.glob("*/")):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            if models and model not in models:
                continue

            for k_dir in sorted(model_dir.glob("k*/")):
                if not k_dir.is_dir():
                    continue
                bench_file = k_dir / "benchmark_results.jsonl"
                if not bench_file.exists() or bench_file.stat().st_size == 0:
                    continue
                try:
                    k = int(k_dir.name[1:])
                except ValueError:
                    continue
                if k_values and k not in k_values:
                    continue
                configs.append((fw, model, k))

    return configs


def eval_exists(framework: str, model: str, k: int, results_dir: str) -> bool:
    path = Path(results_dir) / framework / model / f"k{k}" / "evaluation_results.jsonl"
    return path.exists() and path.stat().st_size > 0


def run_single_eval(framework: str, model: str, k: int, args) -> bool:
    """Run evaluation for one configuration. Returns True on success."""
    cmd = [
        sys.executable, "evaluate_tests.py",
        "--dataset", args.dataset,
        "--framework", framework,
        "--model", model,
        "--k", str(k),
        "--timeout", str(args.timeout),
    ]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.mutation:
        cmd += ["--mutation", "--mutation-timeout", str(args.mutation_timeout)]

    print(f"\n{'─'*60}")
    print(f"  Framework : {framework}")
    print(f"  Model     : {model}")
    print(f"  k         : {k}")
    print(f"  Mutation  : {'yes' if args.mutation else 'no'}")
    print(f"{'─'*60}")

    start = time.perf_counter()
    try:
        per_task = args.mutation_timeout if args.mutation else args.timeout
        total_timeout = per_task * (args.limit or 100) + 120
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            timeout=total_timeout,
        )
        elapsed = time.perf_counter() - start
        if result.returncode == 0:
            print(f"  ✅ Done in {format_duration(elapsed)}")
            return True
        else:
            print(f"  ❌ Failed (exit code {result.returncode}) after {format_duration(elapsed)}")
            return False
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        print(f"  ⏰ Timed out after {format_duration(elapsed)}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Batch evaluate all benchmark results")

    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help="Path to original dataset JSONL")
    parser.add_argument("--results-dir", default="results",
                        help="Root results directory to scan")
    parser.add_argument("--frameworks", nargs="+", default=None,
                        help="Filter: only evaluate these frameworks")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Filter: only evaluate these models")
    parser.add_argument("--k-values", nargs="+", type=int, default=None,
                        help="Filter: only evaluate these k values")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit tasks per config")
    parser.add_argument("--timeout", type=int, default=30,
                        help="Per-task pytest timeout in seconds")
    parser.add_argument("--mutation", action="store_true",
                        help="Enable mutation testing (slow, ~30s per task)")
    parser.add_argument("--mutation-timeout", type=int, default=300,
                        help="Per-task mutation testing timeout in seconds")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip configs that already have evaluation_results.jsonl")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                        help="Re-evaluate all configs even if results exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned evaluations without running")

    return parser.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    configs = discover_configs(
        args.results_dir,
        frameworks=args.frameworks,
        models=args.models,
        k_values=args.k_values,
    )

    if not configs:
        print(f"\n⚠️  No completed benchmark_results.jsonl found in '{args.results_dir}/'")
        print(f"   Run 'python run_all_experiments.py' first to generate results.")
        return

    total = len(configs)

    print(f"\n{'='*60}")
    print(f"  Batch Evaluator")
    print(f"{'='*60}")
    print(f"  Configs found : {total}")
    print(f"  Mutation test : {'yes' if args.mutation else 'no'}")
    print(f"  Skip existing : {args.skip_existing}")
    if args.dry_run:
        print(f"\n  [DRY RUN - no actual execution]\n")
    print(f"{'='*60}")

    if args.dry_run:
        for i, (fw, model, k) in enumerate(configs, 1):
            exists = eval_exists(fw, model, k, args.results_dir)
            status = "⏭  (done)" if exists else "🔲 (pending)"
            print(f"  {i:3d}/{total}  {status}  {fw:12s}  {model:30s}  k={k}")
        return

    passed, failed, skipped = [], [], []
    overall_start = time.perf_counter()

    # Progress bar
    pbar = tqdm(configs, desc="Evaluation Progress", unit="config", ncols=100)
    
    for i, (fw, model, k) in enumerate(pbar, 1):
        # Update progress bar description
        pbar.set_description(f"[{i}/{total}] {fw}/{model}/k={k}")

        if args.skip_existing and eval_exists(fw, model, k, args.results_dir):
            tqdm.write(f"  ⏭  Skipping (evaluation already exists)")
            skipped.append((fw, model, k))
            continue

        ok = run_single_eval(fw, model, k, args)
        if ok:
            passed.append((fw, model, k))
        else:
            failed.append((fw, model, k))
        
        # Update progress bar postfix with stats
        pbar.set_postfix({"✅": len(passed), "❌": len(failed), "⏭": len(skipped)})
    
    pbar.close()

    elapsed = time.perf_counter() - overall_start

    print(f"\n{'='*60}")
    print(f"  Evaluation Complete in {format_duration(elapsed)}")
    print(f"{'='*60}")
    print(f"  ✅ Passed  : {len(passed)}")
    print(f"  ❌ Failed  : {len(failed)}")
    print(f"  ⏭  Skipped : {len(skipped)}")

    if failed:
        print(f"\n  Failed configurations:")
        for fw, model, k in failed:
            print(f"    - {fw} / {model} / k={k}")

    print(f"\n  Next step: python visualize_results_v2.py")


if __name__ == "__main__":
    main()
