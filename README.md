# SVCO Cardiac Output Estimation Project

This repository is a cleaned GitHub-ready package of the codebase from `D:\乱七八糟\董雪毕设\pythonProject`.

## Project Summary

The project focuses on stroke volume (`SV`) and cardiac output (`CO`) estimation from pulse-wave / PPG-like signals. The codebase contains:

- signal preprocessing scripts
- slice generation and structured dataset preparation
- TensorFlow training code with ONNX export
- ONNX inference code
- a PyQt5 desktop monitoring interface
- experimental notebook and older archived scripts

## Repository Layout

```text
.
|-- archive/
|   `-- legacy_scripts/        # historical scripts kept for traceability
|-- data/
|   `-- sample/                # small sample data only
|-- docs/
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

## Main Entry Points

- `src/preprocessing/data_step1.py`: raw signal preprocessing
- `src/preprocessing/data_step2.py`: valley detection and slice generation
- `src/training/train_svco_model.py`: model training and ONNX export
- `src/inference/infer_svco_onnx.py`: single-sample / test-set ONNX inference
- `src/gui/svco_monitor_gui.py`: PyQt5 monitoring GUI
- `packaging/svco.spec`: PyInstaller spec

## Environment

Recommended Python version: `3.10`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Notes Before Running

1. Model weights are not included in this package.
2. Several scripts still contain hard-coded local Windows paths from the original research environment.
3. Some source files use legacy naming and mixed encodings because they were preserved from the original project.
4. The `archive/legacy_scripts/` folder is intentionally retained for comparison and traceability, not as the recommended execution path.

## Suggested Reading Order

1. `docs/repository-structure.md`
2. `src/preprocessing/data_step1.py`
3. `src/preprocessing/data_step2.py`
4. `src/training/train_svco_model.py`
5. `src/inference/infer_svco_onnx.py`
6. `src/gui/svco_monitor_gui.py`

## Data and Privacy

This package includes only a small sample CSV found in the original folder. No trained model artifacts, bundled executables, IDE metadata, or thesis document were included in the packaged repository.
