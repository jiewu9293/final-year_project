"""Aggregate experimental results to compute real numbers for thesis Discussion."""
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path("/Users/jiewudeng/Downloads/fypjiewu/results")
FRAMEWORKS = ["zero_shot", "few_shot", "cot", "tot", "gtot"]
MODELS = [
    "gpt-5.4-mini", "gpt-5.4-nano",
    "claude-sonnet-4-6", "claude-haiku-4-5",
    "deepseek-reasoner", "deepseek-chat",
    "gemini-3-flash-preview",
]
PROVIDER = {
    "gpt-5.4-mini": "OpenAI", "gpt-5.4-nano": "OpenAI",
    "claude-sonnet-4-6": "Anthropic", "claude-haiku-4-5": "Anthropic",
    "deepseek-reasoner": "DeepSeek", "deepseek-chat": "DeepSeek",
    "gemini-3-flash-preview": "Google",
}
TIER = {
    "gpt-5.4-mini": "high", "claude-sonnet-4-6": "high",
    "gpt-5.4-nano": "mid", "deepseek-reasoner": "mid", "deepseek-chat": "mid",
    "claude-haiku-4-5": "low", "gemini-3-flash-preview": "low",
}


def load_eval(framework, model, k):
    path = ROOT / framework / model / f"k{k}" / "evaluation_results.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_bench(framework, model, k):
    path = ROOT / framework / model / f"k{k}" / "benchmark_results.jsonl"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else 0.0


# ============ Main aggregation ============
print("=" * 70)
print("PER-CONFIGURATION AGGREGATE (mean over tasks×samples)")
print("=" * 70)
print(f"{'Framework':<10} {'Model':<25} {'k':<3} {'N':<5} {'pass':<6} {'lcov':<6} {'bcov':<6} {'mut':<6}")

config_summary = {}  # (fw, model, k) -> dict
for fw in FRAMEWORKS:
    for m in MODELS:
        for k in (1, 3, 5):
            rows = load_eval(fw, m, k)
            if not rows:
                continue
            pr = [r.get("pass_rate", 0) for r in rows]
            lc = [r.get("line_coverage", 0) for r in rows]
            bc = [r.get("branch_coverage", 0) for r in rows]
            ms = [r.get("mutation_score", 0) for r in rows]
            # Pass@k: at least one success per task
            by_task = defaultdict(list)
            for r in rows:
                by_task[r["task_id"]].append(r.get("pass_rate", 0))
            pass_at_k = mean([1.0 if max(v) > 0 else 0.0 for v in by_task.values()])
            d = {
                "n_rows": len(rows),
                "n_tasks": len(by_task),
                "pass_rate_mean": safe_mean(pr),
                "line_cov_mean": safe_mean(lc),
                "branch_cov_mean": safe_mean(bc),
                "mut_mean": safe_mean(ms),
                "pass_at_k": pass_at_k,
            }
            config_summary[(fw, m, k)] = d
            print(f"{fw:<10} {m:<25} {k:<3} {len(rows):<5} {d['pass_rate_mean']:.3f}  {d['line_cov_mean']:.3f}  {d['branch_cov_mean']:.3f}  {d['mut_mean']:.3f}")

# ============ By framework ============
print("\n" + "=" * 70)
print("BY FRAMEWORK (averaged across all models, all k)")
print("=" * 70)
print(f"{'Framework':<12} {'pass':<8} {'lcov':<8} {'bcov':<8} {'mut':<8}")
fw_summary = {}
for fw in FRAMEWORKS:
    pr = [d["pass_rate_mean"] for (f, _, _), d in config_summary.items() if f == fw]
    lc = [d["line_cov_mean"] for (f, _, _), d in config_summary.items() if f == fw]
    bc = [d["branch_cov_mean"] for (f, _, _), d in config_summary.items() if f == fw]
    ms = [d["mut_mean"] for (f, _, _), d in config_summary.items() if f == fw]
    fw_summary[fw] = {
        "pass": safe_mean(pr), "lcov": safe_mean(lc),
        "bcov": safe_mean(bc), "mut": safe_mean(ms),
    }
    print(f"{fw:<12} {safe_mean(pr):.3f}    {safe_mean(lc):.3f}    {safe_mean(bc):.3f}    {safe_mean(ms):.3f}")

# ============ By framework × tier ============
print("\n" + "=" * 70)
print("BY FRAMEWORK × MODEL TIER  (mutation score)")
print("=" * 70)
print(f"{'Framework':<12} {'high':<10} {'mid':<10} {'low':<10}")
for fw in FRAMEWORKS:
    by_tier = defaultdict(list)
    for (f, m, k), d in config_summary.items():
        if f == fw:
            by_tier[TIER[m]].append(d["mut_mean"])
    print(f"{fw:<12} {safe_mean(by_tier['high']):.3f}     {safe_mean(by_tier['mid']):.3f}     {safe_mean(by_tier['low']):.3f}")

# ============ By model ============
print("\n" + "=" * 70)
print("BY MODEL (averaged across all frameworks, all k)")
print("=" * 70)
print(f"{'Model':<25} {'tier':<6} {'pass':<8} {'lcov':<8} {'bcov':<8} {'mut':<8}")
model_summary = {}
for m in MODELS:
    pr = [d["pass_rate_mean"] for (_, mm, _), d in config_summary.items() if mm == m]
    lc = [d["line_cov_mean"] for (_, mm, _), d in config_summary.items() if mm == m]
    bc = [d["branch_cov_mean"] for (_, mm, _), d in config_summary.items() if mm == m]
    ms = [d["mut_mean"] for (_, mm, _), d in config_summary.items() if mm == m]
    model_summary[m] = {
        "pass": safe_mean(pr), "lcov": safe_mean(lc),
        "bcov": safe_mean(bc), "mut": safe_mean(ms),
    }
    print(f"{m:<25} {TIER[m]:<6} {safe_mean(pr):.3f}    {safe_mean(lc):.3f}    {safe_mean(bc):.3f}    {safe_mean(ms):.3f}")

# ============ By provider ============
print("\n" + "=" * 70)
print("BY PROVIDER")
print("=" * 70)
print(f"{'Provider':<12} {'pass':<8} {'lcov':<8} {'bcov':<8} {'mut':<8}")
for prov in ["OpenAI", "Anthropic", "DeepSeek", "Google"]:
    pr = [d["pass_rate_mean"] for (_, m, _), d in config_summary.items() if PROVIDER[m] == prov]
    lc = [d["line_cov_mean"] for (_, m, _), d in config_summary.items() if PROVIDER[m] == prov]
    bc = [d["branch_cov_mean"] for (_, m, _), d in config_summary.items() if PROVIDER[m] == prov]
    ms = [d["mut_mean"] for (_, m, _), d in config_summary.items() if PROVIDER[m] == prov]
    print(f"{prov:<12} {safe_mean(pr):.3f}    {safe_mean(lc):.3f}    {safe_mean(bc):.3f}    {safe_mean(ms):.3f}")

# ============ Token consumption per framework ============
print("\n" + "=" * 70)
print("TOKEN CONSUMPTION BY FRAMEWORK (avg total tokens per task)")
print("=" * 70)
print(f"{'Framework':<12} {'mean_tok':<10} {'min':<8} {'max':<8}")
fw_tokens = {}
for fw in FRAMEWORKS:
    all_tok = []
    for m in MODELS:
        for k in (1, 3, 5):
            bench = load_bench(fw, m, k)
            for b in bench:
                t = b.get("total_tokens") or 0
                if t > 0:
                    all_tok.append(t)
    if all_tok:
        fw_tokens[fw] = {"mean": mean(all_tok), "min": min(all_tok), "max": max(all_tok)}
        print(f"{fw:<12} {mean(all_tok):8.0f}   {min(all_tok):<8} {max(all_tok):<8}")
    else:
        print(f"{fw:<12} no benchmark_results data")

# ============ Best configurations ============
print("\n" + "=" * 70)
print("TOP 10 CONFIGURATIONS BY MUTATION SCORE")
print("=" * 70)
ranked = sorted(config_summary.items(), key=lambda kv: -kv[1]["mut_mean"])[:10]
print(f"{'Rank':<5} {'Framework':<12} {'Model':<25} {'k':<3} {'pass':<6} {'mut':<6}")
for i, ((fw, m, k), d) in enumerate(ranked, 1):
    print(f"{i:<5} {fw:<12} {m:<25} {k:<3} {d['pass_rate_mean']:.3f}  {d['mut_mean']:.3f}")

# ============ Save summary JSON ============
out = {
    "config_summary": {f"{fw}|{m}|k{k}": v for (fw, m, k), v in config_summary.items()},
    "framework_summary": fw_summary,
    "model_summary": model_summary,
    "framework_tokens": fw_tokens,
}
with open("/Users/jiewudeng/Downloads/fypjiewu/aggregate_summary.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved: /Users/jiewudeng/Downloads/fypjiewu/aggregate_summary.json")
