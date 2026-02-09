import torch
import os

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 15
LOG_INTERVAL = 100
LR = 1e-3
LATENT_DIM = 32  # Standard for high quality MNIST
KLD_WEIGHT = 1.0  # Standard VAE ELBO
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Configuration
DATASET = 'MNIST'  # Back to MNIST

if DATASET == 'MNIST':
    IMAGE_CHANNELS = 1
    IMAGE_SIZE = 28
elif DATASET == 'CIFAR10':
    IMAGE_CHANNELS = 3
    IMAGE_SIZE = 32
else:
    raise ValueError(f"Unknown dataset: {DATASET}")

# Data Loading
NUM_WORKERS = 4
PIN_MEMORY = True if torch.cuda.is_available() else False
