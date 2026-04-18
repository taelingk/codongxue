# Repository Structure

## What was kept

- core preprocessing scripts
- model training and ONNX export code
- inference and GUI code
- C++ inference examples
- one sample CSV
- notebook and project notes
- legacy scripts moved into `archive/legacy_scripts/`

## What was removed from the GitHub package

- `build/`
- `dist/`
- `output/`
- `.idea/`
- packaged executables
- bundled runtime dependencies
- thesis `.docx`

## Why this package is more GitHub-friendly

- generated binaries are excluded
- source code is grouped by responsibility
- documentation and dependency list are added
- historical scripts are separated from the recommended main path
- the package can be initialized as a clean git repository without carrying local build noise
