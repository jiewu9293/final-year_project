"""
Visualize experiment results with support for multiple frameworks, models, and k values.
Directory structure: figures/{framework}/{model}/k{k}/
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from collections import defaultdict
from matplotlib.lines import Line2D

matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['figure.dpi'] = 150

FIGURES_BASE = Path("figures")
FIGURES_BASE.mkdir(exist_ok=True)


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def merge_data(bench_path, eval_path=None):
    """Merge benchmark and evaluation results by (task_id, sample)."""
    bench_list = load_jsonl(bench_path)
    eval_list = load_jsonl(eval_path) if eval_path else []
    bench = {(r['task_id'], r.get('sample', 1)): r for r in bench_list}
    eval_ = {(r['task_id'], r.get('sample', 1)): r for r in eval_list}
    merged = []
    for key in bench:
        row = {**bench[key], **eval_.get(key, {})}
        merged.append(row)
    return merged


def safe_list(data, key, default=0):
    """Extract a list of values, replacing None with default."""
    return [d.get(key, default) or default for d in data]


def plot_metrics_boxplot(data, figures_dir):
    """Box plot showing distribution of Pass Rate, LCov, BCov, Mut@k."""
    valid = [d for d in data if d.get('pass_rate') is not None]
    if not valid:
        return

    metrics = {
        'Pass Rate':       [d['pass_rate'] for d in valid],
        'Line Coverage':   safe_list(valid, 'line_coverage'),
        'Branch Coverage': safe_list(valid, 'branch_coverage'),
        'Mutation Score':  safe_list(valid, 'mutation_score'),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    bp = ax.boxplot(metrics.values(), patch_artist=True, labels=metrics.keys())
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, (name, vals) in enumerate(metrics.items()):
        mean_val = np.mean(vals)
        ax.text(i + 1, mean_val + 0.03, f'μ={mean_val:.1%}',
                ha='center', fontsize=9, fontweight='bold')

    ax.set_ylim(-0.05, 1.2)
    ax.set_ylabel('Score')
    ax.set_title(f'Distribution of Test Quality Metrics (n={len(valid)} tasks)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / 'metrics_boxplot.png')
    plt.close()
    print(f"  Saved: {figures_dir / 'metrics_boxplot.png'}")


def plot_summary_bar(data, figures_dir):
    """Summary bar chart: mean ± std for each metric."""
    valid = [d for d in data if d.get('pass_rate') is not None]
    if not valid:
        return

    names = ['Pass Rate', 'Line Cov', 'Branch Cov', 'Mut Score']
    means = [
        np.mean([d['pass_rate'] for d in valid]),
        np.mean(safe_list(valid, 'line_coverage')),
        np.mean(safe_list(valid, 'branch_coverage')),
        np.mean(safe_list(valid, 'mutation_score')),
    ]
    stds = [
        np.std([d['pass_rate'] for d in valid]),
        np.std(safe_list(valid, 'line_coverage')),
        np.std(safe_list(valid, 'branch_coverage')),
        np.std(safe_list(valid, 'mutation_score')),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.8)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.03,
                f'{m:.1%}', ha='center', fontweight='bold', fontsize=11)

    ax.set_ylim(0, 1.3)
    ax.set_ylabel('Score (mean ± std)')
    ax.set_title(f'Average Test Quality Metrics (n={len(valid)} tasks)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / 'summary_metrics.png')
    plt.close()
    print(f"  Saved: {figures_dir / 'summary_metrics.png'}")


def plot_energy_vs_quality(data, figures_dir):
    """Scatter plot: energy consumption vs test quality."""
    valid = [d for d in data
             if d.get('energy_kwh_min') and d.get('pass_rate') is not None]
    if not valid:
        return

    energy_mwh = [d['energy_kwh_min'] * 1e6 for d in valid]
    pass_rate = [d['pass_rate'] for d in valid]
    line_cov = safe_list(valid, 'line_coverage')
    mut_score = safe_list(valid, 'mutation_score')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, metric, label, color in zip(
        axes,
        [pass_rate, line_cov, mut_score],
        ['Pass Rate', 'Line Coverage', 'Mutation Score'],
        ['#4C72B0', '#55A868', '#8172B2']
    ):
        ax.scatter(energy_mwh, metric, c=color, s=15, alpha=0.4, edgecolors='none')
        ax.set_xlabel('Energy (mWh)')
        ax.set_ylabel(label)
        ax.set_title(f'Energy vs {label}')
        ax.set_ylim(-0.05, 1.1)
        ax.grid(alpha=0.3)

    plt.suptitle(f'Energy Consumption vs Test Quality (n={len(valid)} tasks)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(figures_dir / 'energy_vs_quality.png')
    plt.close()
    print(f"  Saved: {figures_dir / 'energy_vs_quality.png'}")


def plot_carbon_histogram(data, figures_dir):
    """Histogram of carbon emissions distribution."""
    valid = [d for d in data if d.get('gwp_kgco2eq_min')]
    if not valid:
        return

    gwp_mg = [d['gwp_kgco2eq_min'] * 1e6 for d in valid]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(gwp_mg, bins=30, color='#55A868', alpha=0.7, edgecolor='white')
    mean_val = np.mean(gwp_mg)
    total_val = np.sum(gwp_mg)
    ax.axvline(mean_val, color='black', linestyle='--', linewidth=1.2,
               label=f'Mean={mean_val:.1f} mg CO₂eq')
    ax.set_xlabel('Carbon Emissions per Task (mg CO₂eq)')
    ax.set_ylabel('Number of Tasks')
    ax.set_title(f'Carbon Emission Distribution (n={len(valid)}, total={total_val/1e3:.2f} g CO₂eq)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / 'carbon_histogram.png')
    plt.close()
    print(f"  Saved: {figures_dir / 'carbon_histogram.png'}")


def plot_pass_at_k_comparison(all_results, figures_dir):
    """
    Compare Pass@k across different frameworks and k values.
    all_results: dict of {(framework, model, k): data}
    """
    # Group by framework
    framework_data = defaultdict(dict)
    for (framework, model, k), data in all_results.items():
        # Compute Pass@k for this configuration
        tasks = {}
        for d in data:
            tid = d.get('task_id', '')
            if tid not in tasks:
                tasks[tid] = []
            tasks[tid].append(d)
        
        passed = sum(1 for samples in tasks.values() 
                    if any(s.get('pass_rate', 0) == 1.0 for s in samples))
        pass_at_k = passed / len(tasks) if tasks else 0
        
        framework_data[framework][k] = pass_at_k

    if not framework_data:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'zero_shot': '#4C72B0', 'few_shot': '#55A868', 'gtot': '#C44E52'}
    markers = {'zero_shot': 'o', 'few_shot': 's', 'gtot': '^'}

    for framework, k_values in framework_data.items():
        if not k_values:
            continue
        ks = sorted(k_values.keys())
        pass_rates = [k_values[k] for k in ks]
        
        color = colors.get(framework, '#8172B2')
        marker = markers.get(framework, 'D')
        
        ax.plot(ks, pass_rates, marker=marker, color=color, linewidth=2, 
                markersize=8, label=framework.replace('_', '-'), alpha=0.8)
        
        for k, v in zip(ks, pass_rates):
            ax.annotate(f'{v:.1%}', (k, v), textcoords='offset points',
                       xytext=(0, 10), ha='center', fontsize=9)

    ax.set_xlabel('k (number of samples)', fontsize=12)
    ax.set_ylabel('Pass@k', fontsize=12)
    ax.set_title('Pass@k Comparison Across Frameworks', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / 'pass_at_k_comparison.png')
    plt.close()
    print(f"  Saved: {figures_dir / 'pass_at_k_comparison.png'}")


def plot_framework_comparison(all_results, figures_dir):
    """
    Bar chart comparing frameworks across all k values (k=1,3,5) using subplots.
    """
    from collections import defaultdict
    
    # Group by framework and k
    framework_k_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for (framework, model, k), data in all_results.items():
        valid = [d for d in data if d.get('pass_rate') is not None]
        for d in valid:
            framework_k_data[k][framework]['pass_rate'].append(d.get('pass_rate', 0) or 0)
            framework_k_data[k][framework]['line_coverage'].append(d.get('line_coverage', 0) or 0)
            framework_k_data[k][framework]['branch_coverage'].append(d.get('branch_coverage', 0) or 0)
            framework_k_data[k][framework]['mutation_score'].append(d.get('mutation_score', 0) or 0)
    
    if not framework_k_data:
        return

    # Create subplots for k=1, 3, 5
    k_values = sorted([k for k in framework_k_data.keys() if k in [1, 3, 5]])
    if not k_values:
        return
    
    fig, axes = plt.subplots(1, len(k_values), figsize=(15, 5), sharey=True)
    if len(k_values) == 1:
        axes = [axes]
    
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    metric_names = ['Pass Rate', 'Line Cov', 'Branch Cov', 'Mut Score']
    metric_keys = ['pass_rate', 'line_coverage', 'branch_coverage', 'mutation_score']
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        frameworks = sorted(framework_k_data[k].keys())
        frameworks_display = [fw.replace('_', '-') for fw in frameworks]
        
        # Calculate metrics
        metrics_data = {name: [] for name in metric_names}
        for framework in frameworks:
            for metric_name, metric_key in zip(metric_names, metric_keys):
                metrics_data[metric_name].append(np.mean(framework_k_data[k][framework][metric_key]))
        
        x = np.arange(len(frameworks))
        width = 0.2
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            offset = width * (i - 1.5)
            bars = ax.bar(x + offset, values, width, label=metric if idx == 0 else "", 
                         color=colors[i], alpha=0.8)
            
            for bar, val in zip(bars, values):
                if val > 0.05:  # Only show label if value is significant
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{val:.1%}', ha='center', fontsize=7, rotation=0)
        
        ax.set_xlabel('Framework', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Score', fontsize=11)
        ax.set_title(f'k={k}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(frameworks_display, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
    
    # Add legend only once
    axes[0].legend(fontsize=9, loc='upper left')
    
    fig.suptitle('Framework Comparison Across k Values (averaged across all models)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'framework_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {figures_dir / 'framework_comparison.png'}")


def plot_model_comparison(all_results, figures_dir):
    """
    Grouped bar chart comparing all models across key metrics (averaged over all frameworks and k values).
    """
    model_metrics = defaultdict(lambda: defaultdict(list))

    for (framework, model, k), data in all_results.items():
        valid = [d for d in data if d.get('pass_rate') is not None]
        for d in valid:
            model_metrics[model]['pass_rate'].append(d.get('pass_rate', 0) or 0)
            model_metrics[model]['line_coverage'].append(d.get('line_coverage', 0) or 0)
            model_metrics[model]['branch_coverage'].append(d.get('branch_coverage', 0) or 0)
            model_metrics[model]['mutation_score'].append(d.get('mutation_score', 0) or 0)

    if not model_metrics:
        return

    models = sorted(model_metrics.keys())
    metric_keys = ['pass_rate', 'line_coverage', 'branch_coverage', 'mutation_score']
    metric_labels = ['Pass Rate', 'Line Cov', 'Branch Cov', 'Mut Score']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

    x = np.arange(len(models))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(12, len(models) * 1.5), 6))

    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        means = [np.mean(model_metrics[m][key]) if model_metrics[m][key] else 0 for m in models]
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, means, width, label=label, color=color, alpha=0.8)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.0%}', ha='center', va='bottom', fontsize=7, rotation=90)

    short_names = [m.replace('gpt-5.4', 'gpt5.4').replace('claude-', 'cl-').replace('gemini-', 'gem-').replace('deepseek-', 'ds-') for m in models]
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Cross-Model Comparison (all frameworks & k values)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=25, ha='right', fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = figures_dir / 'model_comparison.png'
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def plot_model_framework_heatmap(all_results, figures_dir):
    """
    Heatmap: rows = models, columns = frameworks, cell = mean mutation score (or pass rate).
    """
    metric = 'mutation_score'
    label = 'Mutation Score'

    cell = defaultdict(lambda: defaultdict(list))
    for (framework, model, k), data in all_results.items():
        for d in data:
            v = d.get(metric)
            if v is not None:
                cell[model][framework].append(v)

    if not cell:
        return

    models = sorted(cell.keys())
    frameworks = sorted({fw for fw, _, _ in all_results.keys()})

    matrix = np.full((len(models), len(frameworks)), np.nan)
    for i, m in enumerate(models):
        for j, fw in enumerate(frameworks):
            vals = cell[m][fw]
            if vals:
                matrix[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(8, len(frameworks) * 1.6), max(5, len(models) * 0.55)))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(range(len(frameworks)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([f.replace('_', '-') for f in frameworks], fontsize=11)
    short_models = [m.replace('gpt-5.4', 'gpt5.4').replace('claude-', 'cl-').replace('gemini-', 'gem-').replace('deepseek-', 'ds-') for m in models]
    ax.set_yticklabels(short_models, fontsize=10)

    for i in range(len(models)):
        for j in range(len(frameworks)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=9, color='black' if 0.3 < val < 0.8 else 'white',
                        fontweight='bold')

    plt.colorbar(im, ax=ax, label=label)
    ax.set_title(f'Model × Framework Heatmap ({label})', fontsize=13, fontweight='bold')
    ax.set_xlabel('Prompting Framework', fontsize=11)
    ax.set_ylabel('Model', fontsize=11)

    plt.tight_layout()
    out = figures_dir / 'model_framework_heatmap.png'
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def plot_cost_vs_quality(all_results, figures_dir):
    """
    Scatter plot: mean input tokens (cost proxy) vs mean mutation score per model.
    Each point = one model, colored by provider.
    """
    PROVIDER_COLORS = {
        'gpt':      '#4C72B0',
        'claude':   '#E07B39',
        'gemini':   '#27AE60',
        'deepseek': '#8E44AD',
    }

    def get_provider(model):
        for p in PROVIDER_COLORS:
            if model.startswith(p):
                return p
        return 'other'

    model_data = defaultdict(lambda: {'tokens': [], 'mutation': [], 'pass_rate': []})
    for (framework, model, k), data in all_results.items():
        for d in data:
            t = d.get('input_tokens') or d.get('last_input_tokens')
            ms = d.get('mutation_score')
            pr = d.get('pass_rate')
            if t and ms is not None:
                model_data[model]['tokens'].append(t)
                model_data[model]['mutation'].append(ms)
            if pr is not None:
                model_data[model]['pass_rate'].append(pr)

    if not model_data:
        print("  Skipping cost_vs_quality: no token data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ['Tokens vs Mutation Score', 'Tokens vs Pass Rate']
    y_keys = ['mutation', 'pass_rate']
    y_labels = ['Mutation Score', 'Pass Rate']

    for ax, title, y_key, y_label in zip(axes, titles, y_keys, y_labels):
        for model, vals in model_data.items():
            if not vals['tokens'] or not vals[y_key]:
                continue
            x = np.mean(vals['tokens'])
            y = np.mean(vals[y_key])
            provider = get_provider(model)
            color = PROVIDER_COLORS.get(provider, '#888888')
            ax.scatter(x, y, color=color, s=120, zorder=5, edgecolors='white', linewidths=0.8)
            short = model.replace('gpt-5.4', 'gpt5.4').replace('claude-', 'cl-').replace('gemini-', 'gem-').replace('deepseek-', 'ds-')
            ax.annotate(short, (x, y), textcoords='offset points', xytext=(6, 4), fontsize=8)

        ax.set_xlabel('Mean Input Tokens (cost proxy)', fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.1)
        ax.grid(alpha=0.3)

    # Legend for providers
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=p.capitalize())
                       for p, c in PROVIDER_COLORS.items()]
    axes[1].legend(handles=legend_elements, fontsize=10, title='Provider')

    plt.suptitle('Token Cost vs Test Quality (per model)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    out = figures_dir / 'cost_vs_quality.png'
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def plot_provider_comparison(all_results, figures_dir):
    """
    Grouped bar chart: aggregate each provider's models, compare OpenAI vs Anthropic vs Google vs DeepSeek.
    """
    PROVIDER_MAP = {
        'gpt':      'OpenAI',
        'claude':   'Anthropic',
        'gemini':   'Google',
        'deepseek': 'DeepSeek',
    }

    def get_provider(model):
        for prefix, name in PROVIDER_MAP.items():
            if model.startswith(prefix):
                return name
        return 'Other'

    provider_metrics = defaultdict(lambda: defaultdict(list))

    for (framework, model, k), data in all_results.items():
        provider = get_provider(model)
        valid = [d for d in data if d.get('pass_rate') is not None]
        for d in valid:
            provider_metrics[provider]['pass_rate'].append(d.get('pass_rate', 0) or 0)
            provider_metrics[provider]['line_coverage'].append(d.get('line_coverage', 0) or 0)
            provider_metrics[provider]['branch_coverage'].append(d.get('branch_coverage', 0) or 0)
            provider_metrics[provider]['mutation_score'].append(d.get('mutation_score', 0) or 0)

    if not provider_metrics:
        return

    providers = ['OpenAI', 'Anthropic', 'Google', 'DeepSeek']
    providers = [p for p in providers if p in provider_metrics]
    metric_keys = ['pass_rate', 'line_coverage', 'branch_coverage', 'mutation_score']
    metric_labels = ['Pass Rate', 'Line Cov', 'Branch Cov', 'Mut Score']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

    x = np.arange(len(providers))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (key, label, color) in enumerate(zip(metric_keys, metric_labels, colors)):
        means = [np.mean(provider_metrics[p][key]) if provider_metrics[p][key] else 0 for p in providers]
        stds = [np.std(provider_metrics[p][key]) if provider_metrics[p][key] else 0 for p in providers]
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                      label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.0%}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Provider', fontsize=12)
    ax.set_ylabel('Score (mean ± std)', fontsize=12)
    ax.set_title('Cross-Provider Comparison (all models, frameworks & k values)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(providers, fontsize=12)
    ax.set_ylim(0, 1.3)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = figures_dir / 'provider_comparison.png'
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def plot_energy_comparison(all_results, figures_dir):
    """
    Compare energy consumption across models and frameworks.
    """
    model_energy = defaultdict(lambda: defaultdict(list))
    
    for (framework, model, k), data in all_results.items():
        for d in data:
            energy = d.get('energy_kwh_min')
            if energy:
                model_energy[model]['energy'].append(energy * 1e6)  # Convert to mWh
                model_energy[model]['quality'].append(d.get('mutation_score', 0) or 0)
    
    if not model_energy:
        print("  Skipping energy_comparison: no energy data found")
        return
    
    models = sorted(model_energy.keys())
    energies = [np.mean(model_energy[m]['energy']) if model_energy[m]['energy'] else 0 for m in models]
    qualities = [np.mean(model_energy[m]['quality']) if model_energy[m]['quality'] else 0 for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Energy consumption bar chart
    colors = ['#4C72B0', '#E07B39', '#27AE60', '#8E44AD', '#C44E52', '#55A868', '#8172B2']
    short_names = [m.replace('gpt-5.4', 'gpt5.4').replace('claude-', 'cl-').replace('gemini-', 'gem-').replace('deepseek-', 'ds-') for m in models]
    
    bars = ax1.bar(range(len(models)), energies, color=colors[:len(models)], alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Energy Consumption (mWh)', fontsize=12)
    ax1.set_title('Average Energy Consumption per Task', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(short_names, rotation=25, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, energies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(energies)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Right: Energy efficiency (quality per mWh)
    efficiency = [q/e if e > 0 else 0 for q, e in zip(qualities, energies)]
    bars2 = ax2.bar(range(len(models)), efficiency, color=colors[:len(models)], alpha=0.8)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Energy Efficiency (Mutation Score / mWh)', fontsize=12)
    ax2.set_title('Energy Efficiency Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(short_names, rotation=25, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, efficiency):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency)*0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    out = figures_dir / 'energy_comparison.png'
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def plot_carbon_comparison(all_results, figures_dir):
    """
    Compare carbon emissions across models and frameworks.
    """
    model_carbon = defaultdict(list)
    
    for (framework, model, k), data in all_results.items():
        for d in data:
            carbon = d.get('gwp_kgco2eq_min')
            if carbon:
                model_carbon[model].append(carbon * 1e6)  # Convert to mg CO2eq
    
    if not model_carbon:
        print("  Skipping carbon_comparison: no carbon data found")
        return
    
    models = sorted(model_carbon.keys())
    mean_carbon = [np.mean(model_carbon[m]) if model_carbon[m] else 0 for m in models]
    total_carbon = [np.sum(model_carbon[m]) if model_carbon[m] else 0 for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#4C72B0', '#E07B39', '#27AE60', '#8E44AD', '#C44E52', '#55A868', '#8172B2']
    short_names = [m.replace('gpt-5.4', 'gpt5.4').replace('claude-', 'cl-').replace('gemini-', 'gem-').replace('deepseek-', 'ds-') for m in models]
    
    # Left: Average carbon per task
    bars1 = ax1.bar(range(len(models)), mean_carbon, color=colors[:len(models)], alpha=0.8)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Carbon Emissions (mg CO₂eq per task)', fontsize=12)
    ax1.set_title('Average Carbon Emissions per Task', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(short_names, rotation=25, ha='right', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars1, mean_carbon):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_carbon)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Right: Total carbon emissions
    total_g = [t/1e3 for t in total_carbon]  # Convert to grams
    bars2 = ax2.bar(range(len(models)), total_g, color=colors[:len(models)], alpha=0.8)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Total Carbon Emissions (g CO₂eq)', fontsize=12)
    ax2.set_title('Total Carbon Emissions (all experiments)', fontsize=13, fontweight='bold')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(short_names, rotation=25, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, total_g):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(total_g)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    out = figures_dir / 'carbon_comparison.png'
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def plot_framework_energy(all_results, figures_dir):
    """
    Compare energy consumption across frameworks.
    """
    framework_energy = defaultdict(list)
    
    for (framework, model, k), data in all_results.items():
        for d in data:
            energy = d.get('energy_kwh_min')
            if energy:
                framework_energy[framework].append(energy * 1e6)  # mWh
    
    if not framework_energy:
        print("  Skipping framework_energy: no energy data found")
        return
    
    frameworks = sorted(framework_energy.keys())
    means = [np.mean(framework_energy[fw]) if framework_energy[fw] else 0 for fw in frameworks]
    stds = [np.std(framework_energy[fw]) if framework_energy[fw] else 0 for fw in frameworks]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#E07B39']
    
    bars = ax.bar(range(len(frameworks)), means, yerr=stds, capsize=5,
                  color=colors[:len(frameworks)], alpha=0.8)
    
    ax.set_xlabel('Framework', fontsize=12)
    ax.set_ylabel('Energy Consumption (mWh per task)', fontsize=12)
    ax.set_title('Energy Consumption by Prompting Framework', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(frameworks)))
    ax.set_xticklabels([f.replace('_', '-') for f in frameworks], fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + max(means)*0.02,
                f'{m:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    out = figures_dir / 'framework_energy.png'
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def main():
    results_base = Path("results")
    
    # Discover all results following pattern: results/{framework}/{model}/k{k}/
    all_results = {}  # {(framework, model, k): data}

    for framework_dir in results_base.glob("*/"):
        if not framework_dir.is_dir():
            continue
        framework = framework_dir.name
        
        for model_dir in framework_dir.glob("*/"):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            
            for k_dir in model_dir.glob("k*/"):
                if not k_dir.is_dir():
                    continue
                k_name = k_dir.name
                k_value = int(k_name[1:])
                
                bench_path = k_dir / "benchmark_results.jsonl"
                eval_path = k_dir / "evaluation_results.jsonl"

                if not bench_path.exists():
                    continue

                data = merge_data(str(bench_path), str(eval_path) if eval_path.exists() else None)
                
                if not data:
                    continue

                print(f"Loading: {framework}/{model}/{k_name} ({len(data)} records)")

                # Store for comparison plots
                all_results[(framework, model, k_value)] = data

    # Generate comparison plots in figures/ root
    if all_results:
        print(f"\n{'='*50}")
        print("Generating comparison plots")
        print(f"{'='*50}")
        # Core figures for paper
        plot_pass_at_k_comparison(all_results, FIGURES_BASE)
        plot_framework_comparison(all_results, FIGURES_BASE)
        plot_model_comparison(all_results, FIGURES_BASE)
        plot_model_framework_heatmap(all_results, FIGURES_BASE)
        plot_cost_vs_quality(all_results, FIGURES_BASE)
        plot_provider_comparison(all_results, FIGURES_BASE)

    print(f"\n✅ All core figures saved to {FIGURES_BASE}/")
    print(f"\nGenerated figures:")
    print(f"  1. pass_at_k_comparison.png       - Pass@k across frameworks")
    print(f"  2. framework_comparison.png       - Framework comparison (k=1)")
    print(f"  3. model_comparison.png           - Model performance comparison")
    print(f"  4. model_framework_heatmap.png    - Model × Framework heatmap")
    print(f"  5. cost_vs_quality.png            - Cost-quality tradeoff")
    print(f"  6. provider_comparison.png        - Cross-provider comparison")


if __name__ == "__main__":
    main()
