# Quant-Singularity: AI-SLM Signal Pod

This repository contains my submission for the Quant Singularity AI-SLM Screening project. It implements a production-ready, Kaggle-compliant machine learning pipeline for generating NIFTY 50 trading signals using a fine-tuned TinyLlama-1.1B model with a deterministic orchestrator and RAG-enabled conviction scoring.

## Kaggle Environment Execution
- **Notebook URL:** https://www.kaggle.com/code/shlokpalrecha/notebook5405391691
- **Hardware:** T4 x2 GPU (Free Tier)

## Repository Structure
- `notebooks/quant_singularity.ipynb`: The complete end-to-end pipeline (training, evaluation, orchestration).
- `lora_adapter/`: The fine-tuned LoRA adapter weights for TinyLlama-1.1B.
- `mlflow_artifacts/mlruns/`: Complete experiment tracking history.
- `report/`: Contains the final written PDF report and evaluation dashboard/CSVs.
- `eval_suite.py`: Pre-committed evaluation suite logic and thresholds.
- `data/`: Placeholder for market states and training JSONL.

## How to Run

1. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. To view the MLflow experiment tracking dashboard locally:
```bash
mlflow ui --backend-store-uri ./mlflow_artifacts/mlruns
```

3. To execute the pipeline, open `notebooks/quant_singularity.ipynb` in a Jupyter environment or Kaggle and run all cells sequentially. The notebook will automatically load the provided datasets, run the LoRA fine-tuning, execute the RAG ablation, and output the orchestrator evaluation suite.
