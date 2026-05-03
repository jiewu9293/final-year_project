# LLM-Based Unit Test Generation: A Comparative Study of Prompting Frameworks

This repository contains the implementation and experimental pipeline for evaluating five prompting frameworks (Zero-shot, Few-shot, Chain-of-Thought, Tree-of-Thought, and Graph-of-Thought) across seven state-of-the-art LLMs for automated unit test generation.

## Overview

This project systematically compares prompting strategies for LLM-based test generation using:
- **5 Prompting Frameworks**: Zero-shot, Few-shot, CoT, ToT, GToT
- **7 LLM Models**: GPT-4o-mini, GPT-4o-nano, Claude-Sonnet-4, Claude-Haiku-4, DeepSeek-Reasoner, DeepSeek-Chat, Gemini-2.0-Flash
- **Evaluation Metrics**: Mutation score, pass rate, line coverage, branch coverage
- **Benchmark**: UnLeakedTestBench (10-task stratified sample)

## Key Features

- Multi-framework prompting implementations with modular design
- Support for OpenAI, Anthropic, DeepSeek, and Google LLM providers
- Comprehensive evaluation pipeline with mutation testing (mutmut)
- Automated experiment orchestration for 105 configurations
- Statistical analysis and visualization tools
- Cost-effectiveness tracking

## Installation

### 1. Create Conda Environment
```bash
conda create -n unleaked python=3.12
conda activate unleaked
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Benchmark Setup (UnLeakedTestBench)

This project uses [UnLeakedTestBench](https://github.com/huangd1999/UnLeakedTestBench) as the evaluation benchmark.

From the project root:
```bash
mkdir -p benchmarks
cd benchmarks
git clone https://github.com/huangd1999/UnLeakedTestBench
```

**Note**: `benchmarks/UnLeakedTestBench/` is intentionally gitignored to keep the repository lightweight.

### 4. API Keys Configuration

Set API keys for the LLM providers you plan to use:

```bash
# OpenAI (GPT-4o-mini, GPT-4o-nano)
export OPENAI_API_KEY="your_openai_key"

# Anthropic (Claude-Sonnet-4, Claude-Haiku-4)
export ANTHROPIC_API_KEY="your_anthropic_key"

# DeepSeek (DeepSeek-Reasoner, DeepSeek-Chat)
export DEEPSEEK_API_KEY="your_deepseek_key"

# Google (Gemini-2.0-Flash)
export GOOGLE_API_KEY="your_google_key"
```

## Usage

### Running a Single Configuration

Generate tests for a specific model-framework-k combination:

```bash
python run_benchmark.py \
  --model gpt-4o-mini \
  --framework cot \
  --k 3 \
  --dataset benchmarks/UnLeakedTestBench/ULT_sample10.jsonl
```

**Parameters**:
- `--model`: Model name (e.g., `gpt-4o-mini`, `claude-sonnet-4`, `deepseek-reasoner`)
- `--framework`: Prompting framework (`zero_shot`, `few_shot`, `cot`, `tot`, `gtot`)
- `--k`: Number of test samples to generate per task (default: 3)
- `--dataset`: Path to the benchmark dataset

### Running All Experiments

Execute the full experimental matrix (105 configurations):

```bash
python run_all_experiments.py
```

This runs all combinations of:
- 7 models × 5 frameworks × 3 k-values (1, 3, 5)

**Note**: This will make extensive API calls and may incur significant costs. Monitor your API usage.

### Evaluation

After test generation, evaluate the results:

```bash
# Evaluate all configurations (without mutation testing)
python evaluate_all.py

# Evaluate with mutation testing (comprehensive but slower)
python evaluate_all.py --mutation
```

This runs:
- **pytest**: Test execution and pass rate
- **coverage.py**: Line and branch coverage
- **mutmut**: Mutation testing for mutation score

### Aggregation and Visualization

Generate summary statistics and figures:

```bash
# Aggregate results into aggregate_summary.json
python aggregate_for_thesis.py

# Generate PDF figures for thesis
python visualize_results_v2.py
```

Outputs:
- `aggregate_summary.json`: Comprehensive results summary
- `figures/result_figures/*.pdf`: Visualizations (framework comparison, model comparison, heatmaps, cost-quality trade-offs, etc.)

## Project Structure

```
.
├── clients/                    # LLM client implementations
│   ├── openai_client.py
│   ├── anthropic_client.py
│   ├── deepseek_client.py
│   └── google_client.py
├── prompting/                  # Prompting framework implementations
│   ├── zero_shot.py
│   ├── few_shot.py
│   ├── cot.py
│   ├── tot.py
│   ├── gtot.py
│   └── templates/              # Prompt templates
├── benchmarks/                 # UnLeakedTestBench (gitignored)
├── results/                    # Generated tests and evaluation results
├── run_benchmark.py            # Single configuration runner
├── run_all_experiments.py      # Full experiment matrix runner
├── evaluate_tests.py           # Core evaluation logic
├── evaluate_all.py             # Batch evaluation script
├── aggregate_for_thesis.py     # Results aggregation
├── visualize_results_v2.py     # Figure generation
└── requirements.txt            # Python dependencies
```

## Supported Models

| Provider   | Model Name              | API Parameter          |
|------------|-------------------------|------------------------|
| OpenAI     | GPT-4o-mini             | `gpt-4o-mini`          |
| OpenAI     | GPT-4o-nano             | `gpt-4o-nano`          |
| Anthropic  | Claude-Sonnet-4         | `claude-sonnet-4`      |
| Anthropic  | Claude-Haiku-4          | `claude-haiku-4`       |
| DeepSeek   | DeepSeek-Reasoner       | `deepseek-reasoner`    |
| DeepSeek   | DeepSeek-Chat           | `deepseek-chat`        |
| Google     | Gemini-2.0-Flash        | `gemini-2.0-flash`     |

## Prompting Frameworks

1. **Zero-shot**: Direct test generation without examples or reasoning steps
2. **Few-shot**: Provides 2 code-test examples before the target
3. **Chain-of-Thought (CoT)**: 4-step reasoning process (understand → plan → generate → verify)
4. **Tree-of-Thought (ToT)**: Multi-path strategy generation and evaluation (2 LLM calls)
5. **Graph-of-Thought (GToT)**: Graph-based code analysis and test synthesis (3 LLM calls)

See `prompting/templates/` for complete prompt templates.

## Results Summary

Key findings from the experimental evaluation:

- **Best Framework**: Tree-of-Thought (ToT) achieved the highest mutation score (67.8%)
- **Best Model**: Claude-Sonnet-4 + ToT (78.3% mutation score)
- **Cost-Effective**: GPT-4o-mini + Few-shot (balanced quality and cost)
- **Provider Ranking**: Anthropic > OpenAI > DeepSeek > Google (by mutation score)

For detailed results, see the thesis document or `aggregate_summary.json`.

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Add delays between requests or reduce batch size in `run_all_experiments.py`
2. **Missing Dependencies**: Ensure all packages in `requirements.txt` are installed
3. **Benchmark Not Found**: Verify `benchmarks/UnLeakedTestBench/` exists and contains `ULT_sample10.jsonl`
4. **Evaluation Failures**: Check that generated tests are syntactically valid Python

### Environment Verification

```bash
# Verify Python version
python --version  # Should be 3.12.x

# Verify key packages
pip list | grep -E "pytest|coverage|mutmut|openai|anthropic"

# Test API connectivity
python -c "import openai; print('OpenAI client ready')"
```

## Citation

If you use this work, please cite:

```
[Your thesis citation information]
```

## License

[Your license information]

## Acknowledgments

- [UnLeakedTestBench](https://github.com/huangd1999/UnLeakedTestBench) for the benchmark dataset
- LLM providers: OpenAI, Anthropic, DeepSeek, Google
