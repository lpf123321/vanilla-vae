import torch
import torch.nn as nn
from torch import Tensor

"""MNIST (28x28) for example"""
class VAE(nn.Module):
    def __init__(self, input_channels=1, image_size=28, latent_dim=32) -> None:
        super(VAE, self).__init__()
        self.encoder = Encoder(image_size, input_channels, latent_dim)
        self.decoder = Decoder(image_size, input_channels, latent_dim)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

class Encoder(nn.Module):
    """Encode input image from pixel space to Gaussian latent space."""
    def __init__(self, image_size = 28, input_channels=1, latent_dim = 32) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1), # 28 -> 14
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 14 -> 7
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.feature_map_size = image_size // 4
        self.flatten_dim = 64 * self.feature_map_size * self.feature_map_size
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        return the mean and variance vector of the Gaussian distribution in the latent space.
        intput
        x: [(batch size, )channels, image_size, image_size]
        return
        mu, logvar: [(batch size, )latent_dim]
        """
        x = self.enc1(x)
        x = self.enc2(x)
        x = x.flatten(start_dim=1)
        return self.fc_mu(x), self.fc_logvar(x)
    
class Decoder(nn.Module):
    """Decode the latent representation z into an image \\hat{x}."""
    def __init__(self, image_size=28, input_channels=1, latent_dim=32) -> None:
        super().__init__()
        self.feature_map_size = image_size // 4
        self.flatten_dim = 64 * self.feature_map_size * self.feature_map_size
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, z: Tensor) -> Tensor:
        """
        z: [(batch size, )latent_dim]
        return x: [(batch size, )channels, image_size, image_size]
        """
        x = self.decoder_input(z)
        x = x.view(-1, 64, self.feature_map_size, self.feature_map_size)
        x = self.dec1(x)
        x = self.dec2(x)
        return x
        
