import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from src.model import VAE
from src import config
from src.trainer import train, test

import warnings
warnings.filterwarnings("ignore", message=".*align should be passed as Python or NumPy boolean.*")

def main():
    if not os.path.exists('data'):
        os.makedirs('data')
    
    if config.DATASET == 'MNIST':
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                                     transform=transforms.ToTensor())
        test_dataset = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())
    elif config.DATASET == 'CIFAR10':
        train_dataset = datasets.CIFAR10('../data', train=True, download=True,
                                       transform=transforms.ToTensor())
        test_dataset = datasets.CIFAR10('../data', train=False, transform=transforms.ToTensor())
    else:
        raise ValueError('Invalid Dataset')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
        
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    model = VAE(input_channels=config.IMAGE_CHANNELS, 
                image_size=config.IMAGE_SIZE,
                latent_dim=config.LATENT_DIM).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    for epoch in range(1, config.EPOCHS + 1):
        train(model, config.DEVICE, train_loader, optimizer, epoch, config.LOG_INTERVAL, config.KLD_WEIGHT)
        test(model, config.DEVICE, test_loader, config.KLD_WEIGHT)
        
        with torch.no_grad():
            sample = torch.randn(64, config.LATENT_DIM).to(config.DEVICE)
            sample = model.decoder(sample).cpu()
            if not os.path.exists('results'):
                os.makedirs('results')
            save_image(sample,
                       'results/sample_' + str(epoch) + '.png')

if __name__ == "__main__":
    main()
