# Sampling On Metric Graphs

## Setup

### Python packages
`pip install -r requirements.txt`

### Build the CUDA kernels

`cd langevin-gpu/src && make`

## Run the experiments

`cd main`

`sorcerun run main.py config.py`

Modify configuration as needed in config.py.

To run grid of experiments with different configurations, use grid-run:

`sorcerun grid-run main.py grid_config.py`

## Relevant code

- `main/main.py` - main script to run the experiments.
- `main/config.py` - configuration file for the experiments.
- `langevin-gpu/src/langevin.cu` - CUDA kernels for Algorithm 1 in the paper.
- `langevin-gpu/python/langevin_simulator.py` - Python wrapper for the CUDA kernels.
- `torch_fvm.py` - PyTorch implementation of the baseline FVM scheme.
