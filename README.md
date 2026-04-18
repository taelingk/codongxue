# SVCO Cardiac Output Estimation

A cleaned and GitHub-ready packaging of the original research codebase for estimating stroke volume (`SV`) and cardiac output (`CO`) from pulse-wave / PPG-like signals.

## Overview

This project contains an end-to-end experimental workflow:

1. preprocess raw pulse-wave signals
2. detect valleys and generate fixed-length slices
3. combine waveform features with clinical features
4. train a TensorFlow model for `SV` prediction
5. export the trained model to ONNX
6. run ONNX inference and visualize results in a PyQt5 desktop GUI

The repository is organized from a previously mixed local research folder into a more standard source layout suitable for GitHub sharing.

## Features

- signal preprocessing for raw CSV waveform data
- valley detection and slice generation
- model training with TensorFlow / Keras
- ONNX export and ONNX Runtime inference
- PyQt5 monitoring interface
- C++ inference reference files
- archived historical scripts for comparison and traceability

## Repository Structure

```text
.
|-- archive/
|   `-- legacy_scripts/        # historical scripts kept as archive material
|-- data/
|   `-- sample/                # small sample data only
|-- docs/
|   |-- repository-structure.md
|   |-- quickstart.md
|   `-- 项目拆解-新手导读.md
|-- notebooks/
|-- packaging/
`-- src/
    |-- cpp/
    |-- gui/
    |-- inference/
    |-- preprocessing/
    |-- tools/
    `-- training/
```

## Recommended Entry Points

- `src/preprocessing/data_step1.py`: raw signal preprocessing
- `src/preprocessing/data_step2.py`: valley detection and slice generation
- `src/training/train_svco_model.py`: model training and ONNX export
- `src/inference/infer_svco_onnx.py`: ONNX inference on single samples or test sets
- `src/gui/svco_monitor_gui.py`: PyQt5 GUI application
- `packaging/svco.spec`: PyInstaller packaging spec

## Typical Workflow

### 1. Preprocess raw waveform data

Use `src/preprocessing/data_step1.py` to:

- read source CSV files
- invert / resample signals
- perform baseline correction
- apply band-pass filtering
- normalize signals

### 2. Generate slices and structured tabular data

Use `src/preprocessing/data_step2.py` to:

- detect valleys
- generate fixed-length signal segments
- attach subject metadata such as age, gender, height, weight, and HR
- export analysis results

### 3. Train the model

Use `src/training/train_svco_model.py` to:

- load structured data
- construct waveform and clinical-feature inputs
- train the ResNet + SE + LSTM model
- save `.keras` artifacts
- export an ONNX model

### 4. Run inference

Use `src/inference/infer_svco_onnx.py` to:

- load the ONNX model
- prepare one-sample or test-set inputs
- predict `SV`
- derive `CO` from `SV` and `HR`

### 5. Launch the GUI

Use `src/gui/svco_monitor_gui.py` for a desktop monitoring workflow with:

- file loading
- waveform display
- trend plotting
- model-based estimation
- result export

## Environment

Recommended Python version: `3.10`

Install dependencies:

```bash
pip install -r requirements.txt
```

Some scripts were originally developed on Windows and assume a desktop Python environment with GUI support.

## Quick Start

### Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Explore the main training code

```bash
python src/training/train_svco_model.py
```

### Explore inference code

```bash
python src/inference/infer_svco_onnx.py
```

### Launch the GUI

```bash
python src/gui/svco_monitor_gui.py
```

For a more practical guide, see `docs/quickstart.md`.

## Data and Artifacts

Included:

- source code
- one sample CSV file in `data/sample/`
- notebook
- packaging spec
- archived historical scripts

Not included:

- trained model weights
- exported ONNX / Keras model artifacts
- local build folders such as `build/`, `dist/`, `output/`
- thesis document files
- IDE metadata

## Known Limitations

1. Model weights are not included in this repository.
2. Several scripts still contain hard-coded Windows paths from the original research environment.
3. Source files retain some original naming and encoding inconsistencies.
4. The archived scripts are preserved for history, not as the recommended primary workflow.
5. The repository has not yet been fully refactored into a reusable Python package.

## Suggested Reading Order

1. `docs/repository-structure.md`
2. `docs/quickstart.md`
3. `src/preprocessing/data_step1.py`
4. `src/preprocessing/data_step2.py`
5. `src/training/train_svco_model.py`
6. `src/inference/infer_svco_onnx.py`
7. `src/gui/svco_monitor_gui.py`

## Notes on License

No explicit open-source license file has been attached yet. Add one only after choosing the intended distribution terms for this project.
