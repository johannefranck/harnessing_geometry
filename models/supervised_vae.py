import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# Encoder
# ---------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                # (B,1,28,28)->(B,784)
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


# ---------------------------------------------------------
# Decoder
# ---------------------------------------------------------
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

    def forward(self, z):
        logits = self.net(z)
        logits = logits.view(-1, 1, 28, 28)
        return logits


# ---------------------------------------------------------
# Supervised VAE (semantic VAE)
# ---------------------------------------------------------
class SupervisedVAE(nn.Module):
    """
    VAE + classifier head on mu (latent means).

    Total loss:
        L = recon + KL + λ * CE(classifier(mu), y)
    """

    def __init__(self, latent_dim: int = 8, num_classes: int = 10, clf_weight: float = 1.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.clf_weight = clf_weight  # λ

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

        # Classifier on latent means
        self.classifier = nn.Linear(latent_dim, num_classes)

    # ---- Encoder helper ----
    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    # ---- Reparameterization ----
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---- Decoder helper ----
    def decode(self, z):
        return self.decoder(z)

    # -----------------------------------------------------
    # Forward: compute supervised VAE loss
    # -----------------------------------------------------
    def forward(self, x, y):
        """
        Inputs:
            x : images (B,1,28,28)
            y : labels  (B,) integer class labels

        Returns:
            loss_total, recon_loss, kl_loss, cls_loss
        """
        # ---- 1. Encode ----
        mu, logvar = self.encode(x)

        # ---- 2. Sample ----
        z = self.reparameterize(mu, logvar)

        # ---- 3. Reconstruction ----
        logits = self.decode(z)

        recon_loss = F.binary_cross_entropy_with_logits(
            logits, x, reduction="sum"
        ) / x.size(0)

        # ---- 4. KL divergence ----
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # ---- 5. Classification loss (semantic term) ----
        logits_y = self.classifier(mu)
        cls_loss = F.cross_entropy(logits_y, y)

        # ---- Total ----
        loss = recon_loss + kl + self.clf_weight * cls_loss

        return loss, recon_loss, kl, cls_loss
