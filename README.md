# Census Income Classifier

Binary income classification (**< $50K** vs. **>= $50K**) and customer segmentation using U.S. Census Bureau data (1994--1995 Current Population Surveys). Built with LightGBM, SHAP, and scikit-learn.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Data Setup](#data-setup)
6. [Running the Code](#running-the-code)
7. [Output Artifacts](#output-artifacts)

---

## Project Overview

A retail business client wants to identify two groups for marketing purposes: people earning **less than $50,000** and those earning **$50,000 or more**, using 40 demographic and employment variables. This project delivers:

1. **Classification model** — A LightGBM gradient-boosted tree classifier, hyperparameter-tuned with Optuna (50-trial Bayesian search with 5-fold stratified CV). Achieves **AUC-ROC 0.957** on a held-out 20% test set.
2. **Segmentation model** — An interpretable depth-4 decision tree trained on the top SHAP features, producing human-readable rules that segment the population into marketing-actionable groups. Accompanied by interactive parallel coordinate visualizations (HiPlot) and global SHAP feature importance analysis.

---

## Repository Structure

```
census-income-classifier/
├── README.md                  # This file
├── pyproject.toml             # Python project config and dependencies
├── uv.lock                    # Locked dependency versions (reproducibility)
├── .python-version            # Python version pin (3.12)
├── .gitignore                 # Git ignore rules
├── ML-TakehomeProject.pdf     # Original project brief
├── census-bureau.columns      # Column header definitions (40 features + weight + label)
├── census-bureau.data         # Dataset (not in repo — see Data Setup below)
│
├── train.py                   # Step 1: Train LightGBM classifier
├── segment.py                 # Step 2: Segmentation analysis (parallel coords + decision tree)
├── shap_importance.py         # Auxiliary: Global SHAP feature importance plot
├── main.py                    # Placeholder entry point
│
├── report/                    # LaTeX project report
│   └── report.tex             # Report source
│
└── artifacts/                 # Generated outputs (created by running scripts)
    ├── best_params.json       # Best Optuna hyperparameters
    ├── feature_importances.csv          # Split-based feature importances
    ├── shap_importances.csv             # SHAP-based feature importances
    ├── shap_summary.png                 # SHAP beeswarm summary plot
    ├── parallel_coordinates.png         # Static parallel coordinates (matplotlib)
    ├── parallel_coordinates.html        # Interactive parallel coordinates (HiPlot)
    ├── decision_tree.png                # Visual decision tree diagram
    ├── decision_tree_rules.txt          # Human-readable segmentation rules
    ├── model.pkl                        # Serialized LightGBM model (git-ignored)
    └── data_splits.pkl                  # Train/test data splits (git-ignored)
```

---

## Prerequisites

- **Python 3.12+**
- **uv** — A fast Python package manager and project tool (replaces pip + venv)

### Installing uv

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Homebrew (macOS):**

```bash
brew install uv
```

**pip (any platform):**

```bash
pip install uv
```

After installation, verify:

```bash
uv --version
```

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/doolingdavid/census-income-classifier.git
cd census-income-classifier
```

2. **Create the virtual environment and install all dependencies:**

```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock`, creates a `.venv/` directory, installs Python 3.12 if needed, and installs all locked dependencies (LightGBM, Optuna, scikit-learn, SHAP, matplotlib, HiPlot, etc.).

That's it. No manual `pip install` needed.

---

## Data Setup

The dataset (`census-bureau.data`, ~92 MB) is too large for Git and must be obtained separately. Place it in the repository root:

```
census-income-classifier/
├── census-bureau.data      ← place here
├── census-bureau.columns   ← already included
└── ...
```

The file should contain 199,523 comma-delimited rows with 42 fields each (40 features + weight + label).

---

## Running the Code

All scripts are run from the repository root using `uv run`, which automatically activates the virtual environment.

### Step 1: Train the classifier

```bash
uv run python train.py
```

This runs the full pipeline:
- Loads and preprocesses the Census Bureau data (199,523 rows, 38 features after dropping non-predictive columns)
- Runs 50-trial Optuna hyperparameter optimization with 5-fold stratified cross-validation (~40 minutes)
- Trains the final LightGBM model with early stopping
- Evaluates on the held-out 20% test set
- Saves all artifacts to `artifacts/`

**Fast re-run** (skips Optuna, uses saved best parameters — ~2 minutes):

```bash
uv run python train.py --skip-optuna
```

This requires `artifacts/best_params.json` from a prior full run.

### Step 2: Generate segmentation analysis

```bash
uv run python segment.py
```

This loads the saved model and data splits, then:
- Computes global SHAP feature importance on 5,000 test samples
- Generates parallel coordinate plots ordered by SHAP importance (both static PNG and interactive HTML)
- Trains a depth-4 decision tree on the top 10 SHAP features
- Saves segmentation rules to `artifacts/decision_tree_rules.txt`

### Step 3 (optional): SHAP feature importance plot

```bash
uv run python shap_importance.py
```

Generates a standalone SHAP beeswarm summary plot and prints the ranked feature importance table.

### Step 4: Compile the project report

```bash
cd report
pdflatex report.tex
pdflatex report.tex   # run twice for cross-references
```

---

## Output Artifacts

After running all scripts, the `artifacts/` directory contains:

| File | Description |
|------|-------------|
| `best_params.json` | Best hyperparameters from Optuna (JSON) |
| `feature_importances.csv` | LightGBM split-based feature importances |
| `shap_importances.csv` | SHAP mean absolute value feature importances |
| `shap_summary.png` | SHAP beeswarm plot (top 20 features) |
| `parallel_coordinates.png` | Static parallel coordinates colored by income class |
| `parallel_coordinates.html` | Interactive HiPlot parallel coordinates (open in browser) |
| `decision_tree.png` | Visual decision tree diagram |
| `decision_tree_rules.txt` | Human-readable segmentation rules |
| `model.pkl` | Serialized LightGBM model (generated locally, not in repo) |
| `data_splits.pkl` | Train/test data splits (generated locally, not in repo) |

The interactive HiPlot visualization (`parallel_coordinates.html`) is a standalone file that can be opened directly in any web browser — no server required.
