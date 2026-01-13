# Vanilla VAE

Implementation of a Variational Autoencoder (VAE) using PyTorch on the MNIST dataset.

## Structure

- `src/`: Source code directory.
  - `model.py`: VAE model architecture (Encoder, Decoder).
  - `config.py`: Hyperparameters and configuration.
  - `trainer.py`: Training and testing loops, loss function.
- `main.py`: Entry point for training and generation.
- `pyproject.toml`: Project configuration and dependencies.

## Usage

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Run training**:
   ```bash
   uv run main.py
   ```

3. **Results**:
   Check `results/` folder for generated samples and reconstruction images.
