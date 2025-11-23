import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.simple_vae import VAE


# --------------------------------------------------------
# Translator model
# --------------------------------------------------------
class TranslatorMLP(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# --------------------------------------------------------
# Discriminator model (latent-space adversary)
# --------------------------------------------------------
class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# --------------------------------------------------------
# Geometry utilities
# --------------------------------------------------------
def pairwise_distances(z: torch.Tensor) -> torch.Tensor:
    diff = z.unsqueeze(1) - z.unsqueeze(0)
    return torch.norm(diff, dim=-1)


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    xx = (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(dim=-1)
    yy = (y.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=-1)
    xy = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=-1)
    gamma = 1.0 / (2 * sigma * sigma)
    return (
        torch.exp(-gamma * xx).mean()
        + torch.exp(-gamma * yy).mean()
        - 2 * torch.exp(-gamma * xy).mean()
    )


# --------------------------------------------------------
# Data loaders
# --------------------------------------------------------
def make_domain_loaders(data_root: str, split_path: str, batch_size: int):
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(
        data_root,
        train=True,
        download=True,
        transform=transform,
    )
    split = torch.load(split_path)
    idxA = split["idxA"]
    idxB = split["idxB"]
    dsA = Subset(full_train, idxA)
    dsB = Subset(full_train, idxB)

    loaderA = DataLoader(
        dsA, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    loaderB = DataLoader(
        dsB, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    return loaderA, loaderB


# --------------------------------------------------------
# Main training function
# --------------------------------------------------------
def train_translators(args):
    device = args.device

    # Load VAEs (frozen)
    vaeA = VAE(latent_dim=args.latent_dim)
    vaeB = VAE(latent_dim=args.latent_dim)
    vaeA.load_state_dict(torch.load(args.weights_vaeA, map_location=device))
    vaeB.load_state_dict(torch.load(args.weights_vaeB, map_location=device))

    vaeA.to(device).eval()
    vaeB.to(device).eval()
    for p in vaeA.parameters():
        p.requires_grad = False
    for p in vaeB.parameters():
        p.requires_grad = False

    # Translators
    T_AB = TranslatorMLP(args.latent_dim).to(device)
    T_BA = TranslatorMLP(args.latent_dim).to(device)

    # Discriminators (latent-domain)
    D_B_disc = LatentDiscriminator(args.latent_dim).to(device)
    D_A_disc = LatentDiscriminator(args.latent_dim).to(device)

    # Optimizers
    opt_T = optim.Adam(
        list(T_AB.parameters()) + list(T_BA.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    opt_D = optim.Adam(
        list(D_B_disc.parameters()) + list(D_A_disc.parameters()),
        lr=args.lr_disc,
    )

    bce = nn.BCELoss()

    # Data loaders
    loaderA, loaderB = make_domain_loaders(
        args.data_root, args.split_path, args.batch_size
    )
    iterA = iter(loaderA)
    iterB = iter(loaderB)

    for step in range(1, args.steps + 1):
        # ---------------------------
        # Load batch (A, B)
        # ---------------------------
        try:
            xA, _ = next(iterA)
        except StopIteration:
            iterA = iter(loaderA)
            xA, _ = next(iterA)

        try:
            xB, _ = next(iterB)
        except StopIteration:
            iterB = iter(loaderB)
            xB, _ = next(iterB)

        xA = xA.to(device)
        xB = xB.to(device)

        with torch.no_grad():
            zA, _ = vaeA.encode(xA)
            zB, _ = vaeB.encode(xB)

        # ---------------------------
        # Forward translations
        # ---------------------------
        zA_B = T_AB(zA)  # A -> B
        zB_A = T_BA(zB)  # B -> A

        # ---------------------------
        # Cycle losses
        # ---------------------------
        zA_cycle = T_BA(zA_B)
        zB_cycle = T_AB(zB_A)

        cycle_loss = (zA_cycle - zA).pow(2).mean() + (zB_cycle - zB).pow(2).mean()

        # ---------------------------
        # Geometry losses
        # ---------------------------
        DA = pairwise_distances(zA)
        DA_m = pairwise_distances(zA_B)
        DB = pairwise_distances(zB)
        DB_m = pairwise_distances(zB_A)

        geom_loss = (DA - DA_m).pow(2).mean() + (DB - DB_m).pow(2).mean()

        # ---------------------------
        # MMD loss (optional)
        # ---------------------------
        if args.w_mmd > 0:
            mmd_loss = mmd_rbf(zA_B, zB, args.mmd_sigma) + mmd_rbf(
                zB_A, zA, args.mmd_sigma
            )
        else:
            mmd_loss = torch.tensor(0.0, device=device)

        # ---------------------------
        # --- Adversarial losses ---
        # ---------------------------
        if args.w_adv > 0:
            # Train discriminators
            opt_D.zero_grad()

            # B-side discriminator: real B vs translated A->B
            real_B = D_B_disc(zB)
            fake_B = D_B_disc(zA_B.detach())
            ones_real_B = torch.ones_like(real_B)
            zeros_fake_B = torch.zeros_like(fake_B)

            loss_D_B = bce(real_B, ones_real_B) + bce(fake_B, zeros_fake_B)

            # A-side discriminator: real A vs translated B->A
            real_A = D_A_disc(zA)
            fake_A = D_A_disc(zB_A.detach())
            ones_real_A = torch.ones_like(real_A)
            zeros_fake_A = torch.zeros_like(fake_A)

            loss_D_A = bce(real_A, ones_real_A) + bce(fake_A, zeros_fake_A)

            loss_D = 0.5 * (loss_D_A + loss_D_B)
            loss_D.backward()
            opt_D.step()

            # Train translators to fool discriminators
            fake_B_for_T = D_B_disc(zA_B)
            fake_A_for_T = D_A_disc(zB_A)

            ones_B_T = torch.ones_like(fake_B_for_T)
            ones_A_T = torch.ones_like(fake_A_for_T)

            adv_loss = 0.5 * (
                bce(fake_B_for_T, ones_B_T) + bce(fake_A_for_T, ones_A_T)
            )
        else:
            adv_loss = torch.tensor(0.0, device=device)

        # ---------------------------
        # Total translator loss
        # ---------------------------
        loss_T = (
            args.w_cycle * cycle_loss
            + args.w_geom * geom_loss
            + args.w_mmd * mmd_loss
            + args.w_adv * adv_loss
        )

        opt_T.zero_grad()
        loss_T.backward()
        opt_T.step()

        if step % args.log_every == 0:
            log = (
                f"[Step {step:5d}] "
                f"Tot={loss_T.item():.4f} "
                f"Cyc={cycle_loss.item():.4f} "
                f"Geo={geom_loss.item():.4f} "
            )
            if args.w_mmd > 0:
                log += f"MMD={mmd_loss.item():.4f} "
            log += f"Adv={adv_loss.item():.4f}"
            print(log)

    torch.save(T_AB.state_dict(), f"checkpoints/{args.out_prefix}_T_AB.pt")
    torch.save(T_BA.state_dict(), f"checkpoints/{args.out_prefix}_T_BA.pt")
    torch.save(D_A_disc.state_dict(), f"checkpoints/{args.out_prefix}_D_A.pt")
    torch.save(D_B_disc.state_dict(), f"checkpoints/{args.out_prefix}_D_B.pt")
    print("Saved translator + discriminator models.")


# --------------------------------------------------------
# CLI
# --------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--weights-vaeA", type=str, default="checkpoints/mnist_split_vaeA.pt")
    ap.add_argument("--weights-vaeB", type=str, default="checkpoints/mnist_split_vaeB.pt")
    ap.add_argument("--split-path", type=str, default="checkpoints/mnist_split_split_indices.pt")
    ap.add_argument("--data-root", type=str, default="data/")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-disc", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--w-cycle", type=float, default=1.0)
    ap.add_argument("--w-geom", type=float, default=0.1)
    ap.add_argument("--w-mmd", type=float, default=0.0)
    ap.add_argument("--w-adv", type=float, default=0.1)
    ap.add_argument("--mmd-sigma", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out-prefix", type=str, default="mnist_split")
    args = ap.parse_args()

    train_translators(args)


if __name__ == "__main__":
    main()
