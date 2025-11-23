import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                # (B, 1, 28, 28) -> (B, 784)
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
        )

    def forward(self, z: torch.Tensor):
        logits = self.net(z)
        logits = logits.view(-1, 1, 28, 28)
        return logits


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x: torch.Tensor):
        """Return posterior parameters (mu, logvar)."""
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        """Return decoder logits."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        """
        Compute standard VAE loss:
          loss = recon_loss (BCE) + KL(q(z|x) || N(0, I))
        Returns: (loss, recon_loss, kl_loss)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)

        # BCE with logits; sum over pixels then average over batch
        recon_loss = F.binary_cross_entropy_with_logits(
            logits, x, reduction="sum"
        ) / x.size(0)

        # KL divergence to N(0, I) in closed form
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        loss = recon_loss + kl
        return loss, recon_loss, kl
