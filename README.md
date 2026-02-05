# Installation Environment Setup
## Create conda environment
```bash
conda create -n unleaked python=3.12
conda activate unleaked
```
## Install required packages
```bash
pip install -r requirements.txt
```
## Benchmark Setup (UnLeakedTestBench)

This project uses [UnLeakedTestBench](https://github.com/huangd1999/UnLeakedTestBench)
 / ULT as an external benchmark.

Download from the project root:
```bash
mkdir -p benchmarks
git clone https://github.com/huangd1999/UnLeakedTestBench benchmarks/UnLeakedTestBench
```
**Note: benchmarks/UnLeakedTestBench/ is intentionally ignored by git to keep the repository lightweight and avoid nesting git repositories.**

## API KEYS
**Set API keys for the LLM providers you plan to run (example for OpenAI):**
```bash
export OPENAI_API_KEY="your_api_key_here"
```
