# Quick Start

## Who this is for

This repository is useful if you want to:

- inspect the original research code structure
- rerun parts of the preprocessing or training flow
- reuse the ONNX inference logic
- study the GUI implementation

## Suggested setup

Use Python `3.10` on Windows.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Recommended order

### Read the project structure first

- `docs/repository-structure.md`
- `docs/项目拆解-新手导读.md`

### Start from preprocessing

Files:

- `src/preprocessing/data_step1.py`
- `src/preprocessing/data_step2.py`

Purpose:

- clean raw waveform data
- detect valleys
- generate fixed-length segments
- build structured samples for later stages

### Then inspect training

File:

- `src/training/train_svco_model.py`

Purpose:

- load prepared data
- construct waveform plus clinical-feature inputs
- train the model
- export ONNX

### Then inspect inference

File:

- `src/inference/infer_svco_onnx.py`

Purpose:

- load the ONNX model
- preprocess single samples
- run ONNX Runtime inference
- compute `CO` from predicted `SV`

### Finally inspect the GUI

File:

- `src/gui/svco_monitor_gui.py`

Purpose:

- load signal files
- run model inference
- show trend charts
- export monitoring results

## Important caveats

1. The repository does not include trained model files.
2. Some scripts still contain absolute local paths from the original machine.
3. The codebase mixes research prototypes and production-like scripts.
4. Historical scripts in `archive/legacy_scripts/` are not the recommended starting point.

## If you want to clean it further

Good next steps:

- replace hard-coded paths with command-line arguments or config files
- standardize file encodings to UTF-8
- split reusable logic into modules
- add test coverage around preprocessing and inference utilities
- choose and add a project license
