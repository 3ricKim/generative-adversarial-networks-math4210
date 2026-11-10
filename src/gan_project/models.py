import torch
from torch import nn


def initialize_dcgan(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def build_models(config):
    if config.synthetic:
        generator = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 128),
                                  nn.ReLU(), nn.Linear(128, 2))
        discriminator = nn.Sequential(nn.Linear(2, 128), nn.LeakyReLU(0.2),
                                      nn.Linear(128, 128), nn.LeakyReLU(0.2),
                                      nn.Linear(128, 1), nn.Flatten(0))
    else:
        # DCGAN tutorial adapted from 64x64 to 32x32: 4 -> 8 -> 16 -> 32.
        generator = nn.Sequential(
            nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, config.channels, 4, 2, 1, bias=False), nn.Tanh(),
        )
        discriminator = nn.Sequential(
            nn.Conv2d(config.channels, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False), nn.Flatten(0),
        )
        generator.apply(initialize_dcgan)
        discriminator.apply(initialize_dcgan)
    return generator, discriminator


def noise(config, n, device, generator=None):
    shape = (n, config.latent_dim) if config.synthetic else (n, 100, 1, 1)
    # CPU generation also permits a separate CPU generator for fixed previews.
    return torch.randn(shape, generator=generator).to(device)
