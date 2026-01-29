# PCIChain Upload Code

This folder contains the code used to run PCIChain experiments, prompts, and evaluation workflows.

## Structure
- `agents/`: agent implementations for the multi-agent pipeline
- `prompts/`: prompt templates and configuration
- `experiments/`: scripts and configs for experiments
- `utils/`: shared utilities
- `chain.py`: core pipeline orchestration
- `rag.py`: retrieval-augmented generation components
- `config.py`: configuration
- `generate_figures.py`: figure generation

## Basic Usage
Adjust settings in `config.py` as needed, then run the pipeline from the project root:

```bash
python chain.py
```

For specific experiments or tests, see the scripts under `experiments/` and `run_deepseek_test.py`.

## Notes
This code is provided as-is for reproducibility of the paper experiments.
