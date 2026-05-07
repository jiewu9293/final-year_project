"""Non-parametric statistical analysis for RQ1 (framework effect on mutation score).

Pipeline:
  1. Load per-sample evaluation_results.jsonl for every (framework, model, k).
  2. For each (framework, task) pair, average mutation_score across all
     models, k-values, and samples -> 10 tasks x 5 frameworks matrix.
  3. Friedman test on the matrix.
  4. Pairwise Wilcoxon signed-rank vs. Zero-shot baseline + Bonferroni
     correction + Cohen's d effect size.
  5. Print results and write a LaTeX-ready table to stats_results.txt.

Run:
    python statistical_analysis.py
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

ROOT = Path(__file__).parent / "myresults" / "results"
FRAMEWORKS = ["zero_shot", "few_shot", "cot", "tot", "gtot"]
MODELS = [
    "gpt-5.4-mini", "gpt-5.4-nano",
    "claude-sonnet-4-6", "claude-haiku-4-5",
    "deepseek-reasoner", "deepseek-chat",
    "gemini-3-flash-preview",
]
K_VALUES = (1, 3, 5)
METRIC = "mutation_score"


def load_rows(framework: str, model: str, k: int) -> list[dict]:
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


def build_matrix() -> tuple[list[str], np.ndarray]:
    """Build a (n_tasks x n_frameworks) matrix of mean mutation scores.

    For each (framework, task) cell we average mutation_score across all
    models, k values and samples. This collapses model/k variability so the
    Friedman test treats each task as a paired observation across frameworks.
    """
    # framework -> task_id -> list of mutation scores
    bucket: dict[str, dict[str, list[float]]] = {fw: defaultdict(list) for fw in FRAMEWORKS}
    for fw in FRAMEWORKS:
        for m in MODELS:
            for k in K_VALUES:
                for r in load_rows(fw, m, k):
                    tid = r.get("task_id")
                    score = r.get(METRIC)
                    if tid is None or score is None:
                        continue
                    bucket[fw][tid].append(float(score))

    # task list = intersection of tasks present in every framework
    task_sets = [set(bucket[fw].keys()) for fw in FRAMEWORKS]
    common_tasks = sorted(set.intersection(*task_sets))
    if not common_tasks:
        raise SystemExit("No tasks shared across all frameworks; cannot run paired tests.")

    matrix = np.array([
        [mean(bucket[fw][t]) for fw in FRAMEWORKS]
        for t in common_tasks
    ])
    return common_tasks, matrix


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Paired Cohen's d using the standard deviation of differences."""
    diff = x - y
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def main() -> None:
    tasks, matrix = build_matrix()
    n_tasks, n_fw = matrix.shape
    print(f"Matrix shape: {n_tasks} tasks x {n_fw} frameworks")
    print(f"Tasks: {tasks}")
    print()

    # Per-framework descriptive stats
    print("Per-framework mutation score (mean across tasks):")
    for j, fw in enumerate(FRAMEWORKS):
        col = matrix[:, j]
        print(f"  {fw:<10} mean={col.mean():.4f}  median={np.median(col):.4f}  sd={col.std(ddof=1):.4f}")
    print()

    # ---- Friedman ----
    cols = [matrix[:, j] for j in range(n_fw)]
    chi2, p_friedman = friedmanchisquare(*cols)
    df = n_fw - 1
    print(f"Friedman: chi2({df}) = {chi2:.3f}, p = {fmt_p(p_friedman)}")
    print()

    # ---- Pairwise Wilcoxon vs Zero-shot ----
    baseline_idx = FRAMEWORKS.index("zero_shot")
    baseline = matrix[:, baseline_idx]
    pairs = [(j, fw) for j, fw in enumerate(FRAMEWORKS) if fw != "zero_shot"]
    n_comp = len(pairs)
    alpha = 0.05
    alpha_bonf = alpha / n_comp

    print(f"Pairwise Wilcoxon vs Zero-shot (Bonferroni alpha = {alpha}/{n_comp} = {alpha_bonf:.4f}):")
    print(f"  {'Framework':<10} {'W':>8} {'p':>10} {'p_bonf':>10} {'d':>8} {'sig':>5}")
    rows_out = []
    for j, fw in pairs:
        x = matrix[:, j]
        try:
            w_stat, p_val = wilcoxon(x, baseline, zero_method="wilcox")
        except ValueError:
            w_stat, p_val = float("nan"), 1.0
        p_bonf = min(1.0, p_val * n_comp)
        d = cohens_d(x, baseline)
        sig = "*" if p_bonf < alpha else ""
        print(f"  {fw:<10} {w_stat:>8.2f} {fmt_p(p_val):>10} {fmt_p(p_bonf):>10} {d:>8.3f} {sig:>5}")
        rows_out.append((fw, w_stat, p_val, p_bonf, d))

    # ---- Save text + LaTeX snippets ----
    out = Path(__file__).parent / "stats_results.txt"
    with open(out, "w") as f:
        f.write(f"Per-task mutation score matrix: {n_tasks} tasks x {n_fw} frameworks\n")
        f.write(f"Tasks: {tasks}\n\n")
        f.write("Per-framework summary (across tasks):\n")
        for j, fw in enumerate(FRAMEWORKS):
            col = matrix[:, j]
            f.write(f"  {fw:<10} mean={col.mean():.4f}  median={np.median(col):.4f}  sd={col.std(ddof=1):.4f}\n")
        f.write(f"\nFriedman: chi2({df}) = {chi2:.3f}, p = {fmt_p(p_friedman)}\n\n")
        f.write("Pairwise Wilcoxon vs Zero-shot (Bonferroni corrected):\n")
        f.write(f"alpha = {alpha}, n_comparisons = {n_comp}, alpha_bonf = {alpha_bonf:.4f}\n\n")
        f.write("LaTeX table:\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\toprule\n")
        f.write("Framework & $W$ & $p$ & $p_{\\text{bonf}}$ & Cohen's $d$ \\\\\n")
        f.write("\\midrule\n")
        label = {"few_shot": "Few-shot", "cot": "CoT", "tot": "ToT", "gtot": "GToT"}
        for fw, w, p, pb, d in rows_out:
            f.write(f"{label.get(fw, fw)} & {w:.2f} & {fmt_p(p)} & {fmt_p(pb)} & {d:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
