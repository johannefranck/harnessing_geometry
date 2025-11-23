import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.simple_vae import VAE


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


def pairwise_distances(z: torch.Tensor) -> torch.Tensor:
    """Compute pairwise Euclidean distances.
    z: (B, M)
    returns: (B, B)
    """
    diff = z.unsqueeze(1) - z.unsqueeze(0)   # (B, B, M)
    D = torch.norm(diff, dim=-1)            # (B, B)
    return D


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute RBF-kernel MMD^2 between two sets of samples x and y.
    x: (B, M)
    y: (B, M)  (assume same batch size for simplicity)
    Uses the biased estimator:
        MMD^2 = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]
    """
    # x, y on same device
    Bx = x.size(0)
    By = y.size(0)

    # Pairwise squared distances
    xx = x.unsqueeze(1) - x.unsqueeze(0)   # (Bx, Bx, M)
    yy = y.unsqueeze(1) - y.unsqueeze(0)   # (By, By, M)
    xy = x.unsqueeze(1) - y.unsqueeze(0)   # (Bx, By, M)

    xx_sq = (xx ** 2).sum(dim=-1)  # (Bx, Bx)
    yy_sq = (yy ** 2).sum(dim=-1)  # (By, By)
    xy_sq = (xy ** 2).sum(dim=-1)  # (Bx, By)

    # RBF kernel
    gamma = 1.0 / (2.0 * sigma * sigma)
    K_xx = torch.exp(-gamma * xx_sq)
    K_yy = torch.exp(-gamma * yy_sq)
    K_xy = torch.exp(-gamma * xy_sq)

    mmd2 = K_xx.mean() + K_yy.mean() - 2.0 * K_xy.mean()
    return mmd2


def make_domain_loaders(data_root: str, split_path: str, batch_size: int):
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(data_root, train=True,
                                download=True, transform=transform)
    split = torch.load(split_path)
    idxA = split["idxA"]
    idxB = split["idxB"]
    dsA = Subset(full_train, idxA)
    dsB = Subset(full_train, idxB)

    loaderA = DataLoader(dsA, batch_size=batch_size, shuffle=True,
                         num_workers=0, pin_memory=False)
    loaderB = DataLoader(dsB, batch_size=batch_size, shuffle=True,
                         num_workers=0, pin_memory=False)
    return loaderA, loaderB


def train_translators(args):
    device = args.device

    # Load VAEs and freeze parameters
    vaeA = VAE(latent_dim=args.latent_dim)
    vaeA.load_state_dict(torch.load(args.weights_vaeA, map_location=device))
    vaeB = VAE(latent_dim=args.latent_dim)
    vaeB.load_state_dict(torch.load(args.weights_vaeB, map_location=device))

    vaeA.to(device).eval()
    vaeB.to(device).eval()
    for p in vaeA.parameters():
        p.requires_grad = False
    for p in vaeB.parameters():
        p.requires_grad = False

    # Translators A->B and B->A
    T_AB = TranslatorMLP(args.latent_dim).to(device)
    T_BA = TranslatorMLP(args.latent_dim).to(device)

    optimizer = optim.Adam(
        list(T_AB.parameters()) + list(T_BA.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Data loaders for both domains using the same split as training
    loaderA, loaderB = make_domain_loaders(
        args.data_root, args.split_path, args.batch_size
    )
    iterA = iter(loaderA)
    iterB = iter(loaderB)

    for step in range(1, args.steps + 1):
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

        # Encode to latent means (no sampling)
        with torch.no_grad():
            muA, logvarA = vaeA.encode(xA)
            muB, logvarB = vaeB.encode(xB)

        zA = muA   # (B, M)
        zB = muB   # (B, M)

        # Translate A -> B and B -> A
        zA_B = T_AB(zA)   # A -> B
        zB_A = T_BA(zB)   # B -> A

        # Cycle A: A -> B -> A
        zA_cycle = T_BA(zA_B)
        cycle_loss_A = torch.mean((zA_cycle - zA) ** 2)

        # Cycle B: B -> A -> B
        zB_cycle = T_AB(zB_A)
        cycle_loss_B = torch.mean((zB_cycle - zB) ** 2)

        cycle_loss = cycle_loss_A + cycle_loss_B

        # Geometry loss via pairwise distances
        D_A = pairwise_distances(zA)
        D_A_mapped = pairwise_distances(zA_B)
        D_B = pairwise_distances(zB)
        D_B_mapped = pairwise_distances(zB_A)

        geom_loss_A = torch.mean((D_A - D_A_mapped) ** 2)
        geom_loss_B = torch.mean((D_B - D_B_mapped) ** 2)
        geom_loss = geom_loss_A + geom_loss_B

        # MMD loss: align translated distributions with real ones
        if args.w_mmd > 0.0:
            mmd_AB = mmd_rbf(zA_B, zB, sigma=args.mmd_sigma)
            mmd_BA = mmd_rbf(zB_A, zA, sigma=args.mmd_sigma)
            mmd_loss = mmd_AB + mmd_BA
        else:
            mmd_loss = torch.tensor(0.0, device=device)

        total_loss = (
            args.w_cycle * cycle_loss
            + args.w_geom * geom_loss
            + args.w_mmd * mmd_loss
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % args.log_every == 0:
            print(
                f"[Step {step:6d}] "
                f"Total={total_loss.item():.4f} "
                f"Cycle={cycle_loss.item():.4f} "
                f"Geom={geom_loss.item():.4f} "
                f"MMD={mmd_loss.item():.4f}"
            )

    torch.save(T_AB.state_dict(), f"checkpoints/{args.out_prefix}_T_AB.pt")
    torch.save(T_BA.state_dict(), f"checkpoints/{args.out_prefix}_T_BA.pt")
    print("Saved translators:",
          args.out_prefix + "_T_AB.pt",
          args.out_prefix + "_T_BA.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--weights-vaeA", type=str, default="checkpoints/mnist_split_vaeA.pt")
    ap.add_argument("--weights-vaeB", type=str, default="checkpoints/mnist_split_vaeB.pt")
    ap.add_argument("--split-path", type=str, default="checkpoints/mnist_split_split_indices.pt")
    ap.add_argument("--data-root", type=str, default="data/")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--w-cycle", type=float, default=1.0)
    ap.add_argument("--w-geom", type=float, default=0.1)
    ap.add_argument("--w-mmd", type=float, default=0.0)      # 0.0 = off by default
    ap.add_argument("--mmd-sigma", type=float, default=1.0)  # RBF kernel width
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--out-prefix", type=str, default="mnist_split")
    args = ap.parse_args()

    train_translators(args)


if __name__ == "__main__":
    main()
