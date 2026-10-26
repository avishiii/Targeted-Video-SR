#!/usr/bin/env bash
# Run the Kaggle notebook from VS Code's terminal, in the background on Kaggle's GPU.
# Needs: pip install kaggle, API token in ~/.kaggle/kaggle.json (Kaggle → Settings → API),
# your username in kernel-metadata.json, and GIT_URL set in the notebook's first cell.
set -e
cd "$(dirname "$0")"
cp ../notebooks/kaggle_run.ipynb .
kaggle kernels push -p .
echo "Status:  kaggle kernels status YOUR_KAGGLE_USERNAME/targeted-vsr-final"
echo "Outputs: kaggle kernels output YOUR_KAGGLE_USERNAME/targeted-vsr-final -p ../kaggle_outputs"
