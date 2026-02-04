Installation
Environment Setup
# Create conda environment
conda create -n unleaked python=3.12

conda activate unleaked
# Install required packages
pip install -r requirements.txt

Benchmark Setup (UnLeakedTestBench)

This project uses UnLeakedTestBench / ULT as an external benchmark.

Download

From the project root:

mkdir -p benchmarks

git clone https://github.com/huangd1999/UnLeakedTestBench benchmarks/UnLeakedTestBench

Note: benchmarks/UnLeakedTestBench/ is intentionally ignored by git to keep the repository lightweight and avoid nesting git repositories.
