# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for blank deformation RIMD CVAE (Conditional Variational Autoencoder). The project appears to be in early development stages with only a virtual environment currently present.

## Development Environment

- **Python Virtual Environment**: Located in `venv/` directory
- **Python Version**: 3.12 (based on venv structure)
- **Activation**: Use `source venv/bin/activate` to activate the virtual environment

## Common Commands

Since this project is in early stages, standard Python development commands would apply:

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (when requirements.txt exists)
pip install -r requirements.txt

# Run Python scripts
python script_name.py

# Install development dependencies
pip install -e .

# Deactivate virtual environment
deactivate
```

## Project Structure

Currently minimal structure:
- `venv/` - Python virtual environment
- Future structure likely to include:
  - Source code for CVAE implementation
  - Data preprocessing scripts
  - Training and evaluation scripts
  - Configuration files
  - Documentation

## Notes for Future Development

- Project name suggests focus on blank deformation using RIMD CVAE architecture
- Will likely involve PyTorch/TensorFlow for deep learning implementation
- May include data loaders, model definitions, training loops, and evaluation metrics
- Consider adding requirements.txt, setup.py, and proper project structure as development progresses
