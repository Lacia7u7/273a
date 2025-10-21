# Heterogeneous GNN Readmission Pipeline

This repository contains a production-oriented implementation of a heterogeneous graph neural network pipeline for predicting 30-day readmission using the Diabetes 130-US hospitals dataset.

## Getting Started

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter Lab/Notebook and open `main.ipynb`.
4. Edit the JSON configuration cell to point to the dataset and mapping files.
5. Execute the notebook sequentially.

Artifacts (models, scalers, vocabs, metrics, plots, TensorBoard logs) are stored in the configured artifacts directory.

Unit tests cover vocabulary handling, graph construction, and split leakage.

